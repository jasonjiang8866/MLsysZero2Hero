"""Stage the E01 workload and independently verify retrieved CUDA traces on CPU."""

import array
import base64
import hashlib
import io
import json
import math
import shlex
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Annotated

from pydantic import Field

from .e01 import CASES, CHUNKS, DELAY, SIZE, STEPS, kernel_timeline
from .results import DIGEST, Strict
from .worker import Runtime

# pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime, linux/amd64
IMAGE_DIGEST = "sha256:f0ca81b440e252399d9954a45b616ee2540959466aacf3dfc3f856691eee66e8"
FILES = {"summary.json", *(f"{case}.json" for case in CASES)}


class TraceFile(Strict):
    bytes: Annotated[int, Field(gt=0, le=32 * 1024**2)]
    sha256: DIGEST


class TraceExport(Strict):
    runtime: Runtime
    files: dict[str, TraceFile]


def launch_options(image_digest):
    if image_digest != IMAGE_DIGEST:
        raise ValueError("e01_image_digest_mismatch")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("mlsys_lab/__init__.py", b"")
        for name in ("e01.py", "e01_cloud_worker.py", "result_server.py"):
            bundle.writestr("mlsys_lab/" + name, Path(__file__).with_name(name).read_bytes())
    encoded = base64.b64encode(buffer.getvalue()).decode()
    # Only these course files travel to the Pod. The bootstrap needs no install or checkout.
    command = (
        "import base64,os,sys; p='/tmp/e01-course.zip'; "
        f"open(p,'wb').write(base64.b64decode('{encoded}')); "
        "os.environ['PYTHONPATH']=p; os.execv(sys.executable,"
        "[sys.executable,'-m','mlsys_lab.e01_cloud_worker'])"
    )
    return {
        "disk": 30,
        "entrypoint": ["/bin/bash", "-lc"],
        "cmd": ["exec " + shlex.join(["python", "-c", command])],
    }


def reference_digest():
    digest = hashlib.sha256()
    for step in range(STEPS):
        row = array.array("f", range(step, SIZE + step))
        if sys.byteorder != "little":
            row.byteswap()
        digest.update(row.tobytes())
    return digest.hexdigest()


def verify(root, envelope):
    for name, metadata in envelope.files.items():
        raw = (root / name).read_bytes()
        if len(raw) != metadata.bytes or hashlib.sha256(raw).hexdigest() != metadata.sha256:
            raise ValueError("e01_file_hash_mismatch")
    report = json.loads((root / "summary.json").read_bytes())
    expected = reference_digest()
    inputs = array.array("f", range(SIZE))
    if sys.byteorder != "little":
        inputs.byteswap()
    if (
        report["experiment"] != "E01"
        or set(report["results"]) != set(CASES)
        or report["reference_sha256"] != expected
        or report["input_sha256"] != hashlib.sha256(inputs.tobytes()).hexdigest()
        or report["device_count"] != 1
        or report["gpu"] != envelope.runtime.job.gpu
        or report["steps"] != STEPS
        or report["elements_per_step"] != SIZE
        or report["dtype"] != "torch.float32"
        or report["chunks"] != CHUNKS
        or report["producer_delay_ms"] != DELAY * 1000
        or report["cuda"] != "12.8"
        or report["torch"].split("+")[0] != "2.9.0"
    ):
        raise ValueError("e01_summary_mismatch")
    for case in CASES:
        row = report["results"][case]
        trace = json.loads((root / f"{case}.json").read_bytes())
        events = trace["traceEvents"]
        timeline = kernel_timeline(events)
        annotations = [e["name"] for e in events if e.get("ph") == "X" and "name" in e]
        count = STEPS * (CHUNKS if case == "small_launches" else 1)
        if (
            row["trace"] != f"{case}.json"
            or row["output_sha256"] != expected
            or any(v != row[k] for k, v in timeline.items())
            or timeline["kernels"] != count
            or annotations.count(case + "/step") != STEPS
            or annotations.count(case + "/producer")
            != (STEPS if case in {"host_cpu", "io_wait"} else 0)
        ):
            raise ValueError("e01_trace_mismatch")
        samples = row["samples"]
        metrics = ("host_submit_ms", "host_thread_cpu_ms", "wall_ms", "stream_span_ms")
        if len(samples) != len(CASES) or not all(
            math.isfinite(s[key]) and s[key] > 0 for s in samples for key in metrics
        ):
            raise ValueError("e01_timing_invalid")
        median = statistics.median(s["wall_ms"] for s in samples)
        if row["median_wall_ms"] != median or row["steps_per_second"] != STEPS * 1000 / median:
            raise ValueError("e01_median_invalid")


def unpack(raw, root, runtime):
    """Allow only the five expected data files and their runtime-bound manifest."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            names = bundle.namelist()
            if len(names) != 6 or set(names) != FILES | {"manifest.json"}:
                raise ValueError("e01_archive_files_invalid")
            if bundle.getinfo("manifest.json").file_size > 65536:
                raise ValueError("e01_manifest_too_large")
            manifest = bundle.read("manifest.json")
            envelope = TraceExport.model_validate_json(manifest)
            if envelope.runtime != runtime or set(envelope.files) != FILES:
                raise ValueError("export_identity_invalid")
            for name, metadata in envelope.files.items():
                if bundle.getinfo(name).file_size != metadata.bytes:
                    raise ValueError("e01_archive_size_mismatch")
                (root / name).write_bytes(bundle.read(name))
    except zipfile.BadZipFile as exc:
        raise ValueError("e01_invalid_zip") from exc
    verify(root, envelope)
    (root / "manifest.json").write_bytes(manifest)
    return manifest
