"""CPU replay checks for the hosted forward RMSNorm bundle."""

import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

from . import e02
from .e01 import kernel_timeline

FILES = {"summary.json", *e02.trace_names()}


def verify(root, envelope):
    for name, metadata in envelope.files.items():
        data = (root / name).read_bytes()
        if len(data) != metadata.bytes or hashlib.sha256(data).hexdigest() != metadata.sha256:
            raise ValueError("e01_file_hash_mismatch")
    report = json.loads((root / "summary.json").read_bytes())
    expected = {
        "experiment": "E02-RMSNorm-forward",
        "seed": e02.SEED,
        "epsilon": e02.EPSILON,
        "repeats": e02.REPEATS,
        "calls_per_sample": e02.BATCH,
        "device_count": 1,
        "cuda": "12.8",
        "triton": "3.5.0",
        "gpu": envelope.runtime.job.gpu,
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ["e02.py", "e02_kernel.py"]
        },
    }
    if any(report[k] != v for k, v in expected.items()) or report["torch"].split("+")[0] != "2.9.0":
        raise ValueError("e02_summary_mismatch")
    matrix = [
        (dtype, rows, width, step) for dtype in e02.TOLERANCES for rows, width, step in e02.CASES
    ]
    inputs = {}
    for case, (dtype, rows, width, step) in zip(report["cases"], matrix, strict=True):
        tol = e02.TOLERANCES[dtype]
        hashes = (case["input_sha256"], case["weight_sha256"])
        if (
            case["dtype"] != dtype
            or case["shape"] != [rows, width]
            or case["strides"] != [width * step, step]
            or case["atol"] != tol
            or case["rtol"] != tol
            or set(case["methods"]) != {"eager", "fused"}
            or not all(re.fullmatch("[0-9a-f]{64}", h) for h in hashes)
            or inputs.setdefault((dtype, rows, width), hashes) != hashes
        ):
            raise ValueError("e02_case_mismatch")
        for method, result in case["methods"].items():
            name = f"{dtype}-{rows}x{width}-s{step}-{method}.json"
            samples = result["samples_us"]
            events = json.loads((root / name).read_bytes())["traceEvents"]
            kernels = kernel_timeline(events)["kernels"]
            if (
                result["trace"] != name
                or result["trace_sha256"] != envelope.files[name].sha256
                or result["kernels"] != kernels
                or (method == "fused" and kernels != 1)
                or len(samples) != e02.REPEATS
                or not all(math.isfinite(s) and s > 0 for s in samples)
                or result["median_us"] != median(samples)
                or not 0 <= result["max_tolerance_ratio"] <= 1
                or not math.isfinite(result["max_abs_error"])
                or result["max_abs_error"] < 0
            ):
                raise ValueError("e02_trace_mismatch")
