"""Fixed E00 worker and result-only server; never creates or deletes cloud resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .cli import run_smoke
from .result_server import State, server
from .results import SHA, Manifest, Strict, bounded, payloads, verify

IMAGE = "ghcr.io/jasonjiang8866/mlsyszero2hero-worker@"
GPU = Literal[
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
]


class Job(Strict):
    schema_version: Literal[1] = 1
    repository: Literal["jasonjiang8866/MLsysZero2Hero"] = "jasonjiang8866/MLsysZero2Hero"
    experiment: Literal["E00", "E01"] = "E00"
    attempt: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")]
    source_sha: SHA
    image_digest: Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
    gpu: GPU
    runtime_seconds: Annotated[int, Field(ge=1, le=60)] = 30
    collect_seconds: Annotated[int, Field(ge=1, le=300)] = 120

    @property
    def image(self) -> str:
        if self.experiment == "E01":
            return "pytorch/pytorch@" + self.image_digest
        return IMAGE + self.image_digest


class Runtime(Strict):
    job: Job
    pod_id: Annotated[str, Field(pattern=r"^[a-z0-9]{1,64}$")]

    def fingerprint(self) -> str:
        raw = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


class ExportManifest(Strict):
    runtime: Runtime
    bundle: Manifest


def admit(root: Path, runtime: Runtime) -> Manifest:
    manifest = verify(root, runtime.job.source_sha, runtime.job.image_digest)
    device = json.loads(bounded(root / "metrics.json"))["device"]
    if device["device_name"] != runtime.job.gpu or device["memory_bytes"] < 24_000_000_000:
        raise ValueError("runtime_gpu_mismatch")
    return manifest


def export(root: Path, runtime: Runtime) -> dict[str, bytes]:
    manifest = admit(root, runtime)
    envelope = ExportManifest(runtime=runtime, bundle=manifest)
    return {
        "/v1/manifest": envelope.model_dump_json().encode(),
        **{"/v1/files/" + k: v for k, v in payloads(root).items()},
    }


def compute(runtime: Runtime, root: Path, build_sha: str, state: State) -> None:
    state.update("running")
    try:
        if runtime.job.source_sha != build_sha:
            raise ValueError("worker_source_mismatch")
        if any(os.environ.get(k) for k in ("GITHUB_TOKEN", "GH_TOKEN", "RCB_LEDGER_TOKEN")):
            raise ValueError("management_credential_present")
        run_smoke(root, build_sha, runtime.job.image_digest, timeout=runtime.job.runtime_seconds)
        state.update("ready", export(root, runtime))
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired):
        state.update("failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/workspace/e00"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        raw = os.environ.pop("MLS_JOB_JSON", "")
        if len(raw) > 8192 or not 1024 <= args.port <= 65535:
            raise ValueError("worker_configuration_invalid")
        runtime = Runtime(job=Job.model_validate_json(raw), pod_id=os.environ["RUNPOD_POD_ID"])
        token = os.environ.pop("MLS_RESULT_TOKEN", "")
        # RunPod may inject a Pod-scoped key. Discard unused; account keys must
        # never be included in the creation request. No management client here.
        os.environ.pop("RUNPOD_API_KEY", None)
        build_sha = Path("/opt/mlsys/SOURCE_SHA").read_text().strip()
        if build_sha != runtime.job.source_sha:
            raise ValueError("worker_source_mismatch")
        state = State()
        with server(runtime, state, token, ("0.0.0.0", args.port)) as http:
            http.timeout = 0.5
            deadline = time.monotonic() + runtime.job.runtime_seconds + runtime.job.collect_seconds
            threading.Thread(
                target=compute, args=(runtime, args.output, build_sha, state), daemon=True
            ).start()
            while time.monotonic() < deadline:
                http.handle_request()
        phase = json.loads(state.response("/v1/status", runtime)[1])["phase"]
        return 0 if phase == "ready" else 2
    except (ValueError, OSError, KeyError, TypeError):
        print('{"status":"worker_refused","kernel_verified":false}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
