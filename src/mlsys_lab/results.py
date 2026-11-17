"""Bounded, identity-bound E00 bundles; numerical pass is not cloud acceptance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .cuda_smoke import PTX, N, check_output, inputs

SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
DIGEST = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class FileHash(Strict):
    bytes: Annotated[int, Field(gt=0, le=65536)]
    sha256: DIGEST


class Manifest(Strict):
    schema_version: Literal[1] = 1
    experiment: Literal["E00"] = "E00"
    source_sha: SHA
    image_digest: Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")] | None
    # Declared revision binding, not proof of a container's actual build provenance.
    kernel_verified: Literal[True] = True
    cloud_acceptance: Literal["not_evaluated"] = "not_evaluated"
    files: dict[str, FileHash]


def bounded(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("artifact_invalid")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("artifact_too_large")
    return raw


def payloads(root: Path) -> dict[str, bytes]:
    if root.is_symlink():
        raise ValueError("bundle_symlink")
    return {name: bounded(root / name) for name in ("metrics.json", "output.json")}


def verify_payloads(files: dict[str, bytes]) -> None:
    metrics = json.loads(files["metrics.json"])
    values = json.loads(files["output.json"])
    if not isinstance(metrics, dict):
        raise ValueError("metrics_invalid")
    if not isinstance(values, list) or any(type(v) not in (int, float) for v in values):
        raise ValueError("output_invalid")
    check_output(values)
    if (
        metrics.get("kernel") != "ptx_vector_add_fp32_v1"
        or metrics.get("elements") != N
        or type(metrics.get("max_abs_error")) not in (int, float)
        or metrics["max_abs_error"] != 0
        or metrics.get("kernel_sha256") != hashlib.sha256(PTX).hexdigest()
        or metrics.get("input_sha256") != hashlib.sha256(json.dumps(inputs()).encode()).hexdigest()
        or metrics.get("timing") != "not_a_performance_benchmark"
        or metrics.get("framework_runtime") != "not_tested"
    ):
        raise ValueError("metrics_invalid")
    device = metrics.get("device", {})
    # Broker schema validates full observed GPU facts, including finite/typed values.
    from runpod_ci_broker.worker import Observation

    observation = Observation.model_validate(device)
    if (
        observation.device_count != 1
        or observation.driver_cuda_version < 12080
        or observation.memory_roundtrip_passed is not True
    ):
        raise ValueError("device_invalid")


def seal(root: Path, source_sha: str, image_digest: str | None) -> Manifest:
    files = payloads(root)
    verify_payloads(files)
    manifest = Manifest(
        source_sha=source_sha,
        image_digest=image_digest,
        files={
            name: FileHash(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            for name, raw in files.items()
        },
    )
    # Exclusive create prevents a failed rerun from rewriting old successful evidence.
    with (root / "manifest.json").open("x") as stream:
        stream.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def verify(root: Path, expected_source: str, expected_image: str | None) -> Manifest:
    if not re.fullmatch(r"[a-f0-9]{40}", expected_source):
        raise ValueError("expected_source_required")
    if root.is_symlink():
        raise ValueError("bundle_symlink")
    manifest = Manifest.model_validate_json(bounded(root / "manifest.json"))
    if manifest.source_sha != expected_source or manifest.image_digest != expected_image:
        raise ValueError("bundle_provenance_mismatch")
    files = payloads(root)
    if set(manifest.files) != set(files):
        raise ValueError("bundle_files_mismatch")
    for name, raw in files.items():
        if (
            manifest.files[name].bytes != len(raw)
            or manifest.files[name].sha256 != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("artifact_hash_mismatch")
    verify_payloads(files)
    return manifest
