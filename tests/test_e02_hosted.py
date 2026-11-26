import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from mlsys_lab import e01_hosted, e02
from mlsys_lab.transport import collect
from mlsys_lab.worker import Job, Runtime


def test_archive_replay_and_reject_tolerance_failure(tmp_path):
    runtime = Runtime(
        pod_id="fixture",
        job=Job(
            experiment="E02",
            attempt="E02-1-1",
            gpu="NVIDIA GeForce RTX 4090",
            source_sha="a" * 40,
            image_digest=e01_hosted.IMAGE_DIGEST,
        ),
    )
    payload = {}
    summary = {
        "experiment": "E02-RMSNorm-forward",
        "seed": 42,
        "epsilon": 1e-5,
        "repeats": 7,
        "calls_per_sample": 20,
        "device_count": 1,
        "cuda": "12.8",
        "triton": "3.5.0",
        "torch": "2.9.0+cu128",
        "gpu": runtime.job.gpu,
        "cases": [],
        "source_sha256": {
            name: hashlib.sha256(Path(e02.__file__).with_name(name).read_bytes()).hexdigest()
            for name in ["e02.py", "e02_kernel.py"]
        },
    }
    for dtype, tol in e02.TOLERANCES.items():
        for rows, width, step in e02.CASES:
            methods = {}
            for method, n in [("fused", 1), ("eager", 4)]:
                name = f"{dtype}-{rows}x{width}-s{step}-{method}.json"
                events = [{"cat": "kernel", "ph": "X", "ts": i, "dur": 0.5} for i in range(n)]
                payload[name] = json.dumps({"traceEvents": events}).encode()
                methods[method] = {
                    "trace": name,
                    "trace_sha256": hashlib.sha256(payload[name]).hexdigest(),
                    "kernels": n,
                    "samples_us": [1.0] * 7,
                    "median_us": 1.0,
                    "max_abs_error": 0.0,
                    "max_tolerance_ratio": 0.0,
                }
            summary["cases"].append(
                {
                    "dtype": dtype,
                    "shape": [rows, width],
                    "strides": [width * step, step],
                    "atol": tol,
                    "rtol": tol,
                    "input_sha256": "a" * 64,
                    "weight_sha256": "b" * 64,
                    "methods": methods,
                }
            )

    def archive():
        buf = io.BytesIO()
        payload["summary.json"] = json.dumps(summary).encode()
        with zipfile.ZipFile(buf, "w") as z:
            files = {}
            for name, data in payload.items():
                z.writestr(name, data)
                files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            z.writestr(
                "manifest.json", json.dumps({"runtime": runtime.model_dump(), "files": files})
            )
        return buf.getvalue()

    class Reader:
        def open(self, path, timeout):
            return io.BytesIO(archive())

    collect(runtime, Reader(), tmp_path / "accepted")
    assert len(list((tmp_path / "accepted").iterdir())) == 23
    summary["cases"][-1]["methods"]["fused"]["max_tolerance_ratio"] = 1.1
    failed = tmp_path / "failed"
    failed.mkdir()
    with pytest.raises(ValueError, match="e02_trace_mismatch"):
        e01_hosted.unpack(archive(), failed, runtime)
