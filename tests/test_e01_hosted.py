import array
import hashlib
import json
from types import SimpleNamespace

import pytest

from mlsys_lab import e01, e01_hosted


def test_four_trace_acceptance_rejects_swapped_profiles(tmp_path):
    payload = {}
    expected = e01_hosted.reference_digest()
    report = {
        "experiment": "E01",
        "gpu": "fixture",
        "steps": e01.STEPS,
        "elements_per_step": e01.SIZE,
        "device_count": 1,
        "dtype": "torch.float32",
        "chunks": e01.CHUNKS,
        "producer_delay_ms": 3,
        "torch": "2.9.0",
        "cuda": "12.8",
        "input_sha256": hashlib.sha256(array.array("f", range(e01.SIZE)).tobytes()).hexdigest(),
        "reference_sha256": expected,
        "results": {},
    }
    for case in e01.CASES:
        n = e01.STEPS * (e01.CHUNKS if case == "small_launches" else 1)
        events = [{"ph": "X", "cat": "kernel", "ts": i * 2, "dur": 1} for i in range(n)]
        events += [{"ph": "X", "name": case + "/step"}] * e01.STEPS
        if case in {"host_cpu", "io_wait"}:
            events += [{"ph": "X", "name": case + "/producer"}] * e01.STEPS
        payload[case + ".json"] = json.dumps({"traceEvents": events}).encode()
        sample = dict.fromkeys(
            ["host_submit_ms", "wall_ms", "host_thread_cpu_ms", "stream_span_ms"], 1
        )
        report["results"][case] = {
            "trace": case + ".json",
            "samples": [sample] * 4,
            "median_wall_ms": 1,
            "steps_per_second": e01.STEPS * 1000,
            "output_sha256": expected,
            **e01.kernel_timeline(events),
        }
    payload["summary.json"] = json.dumps(report).encode()

    def check():
        files = {}
        for name, raw in payload.items():
            (tmp_path / name).write_bytes(raw)
            files[name] = SimpleNamespace(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        job = SimpleNamespace(gpu="fixture")
        e01_hosted.verify(tmp_path, SimpleNamespace(files=files, runtime=SimpleNamespace(job=job)))

    check()
    payload["host_cpu.json"] = payload["io_wait.json"]
    with pytest.raises(ValueError, match="trace_mismatch"):
        check()
