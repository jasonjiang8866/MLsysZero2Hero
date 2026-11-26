"""E02 RMSNorm forward lab. Run python -m mlsys_lab.e02 --help for usage."""

import argparse
import hashlib
import json
import platform
from functools import partial
from pathlib import Path
from statistics import median

from .e01 import kernel_timeline

# Each adjacent comparison changes one variable relative to (64, 1024, 1).
CASES = ((1, 1024, 1), (64, 1024, 1), (512, 1024, 1), (64, 1000, 1), (64, 1000, 2))
TOLERANCES = {"float32": 1e-5, "float16": 1e-3}
EPSILON = 1e-5
SEED = 42
REPEATS = 7
BATCH = 20


def trace_names():
    return {
        f"{dtype}-{rows}x{width}-s{step}-{method}.json"
        for dtype in TOLERANCES
        for rows, width, step in CASES
        for method in ("eager", "fused")
    }


def inputs(torch, rows, width, step, dtype, device):
    gen = torch.Generator().manual_seed(SEED)
    values = torch.randn(rows, width, generator=gen).to(dtype=dtype, device=device)
    backing = torch.full((rows, width * step), float("nan"), dtype=dtype, device=device)
    view = backing[:, ::step]
    view.copy_(values)
    weight = torch.randn(width, generator=gen).to(dtype=dtype, device=device)
    return view, weight


def eager(torch, x, weight):
    values = x.float()
    return (
        values * torch.rsqrt(values.square().mean(dim=1, keepdim=True) + EPSILON) * weight.float()
    ).to(x.dtype)


def fused(torch, x, weight):
    import triton

    from .e02_kernel import rmsnorm_kernel

    rows, width = x.shape
    out = torch.empty((rows, width), dtype=x.dtype, device=x.device)
    rmsnorm_kernel[(rows,)](
        x, weight, out, width, *x.stride(), EPSILON, BLOCK=triton.next_power_of_2(width)
    )
    return out


def compare(torch, result, expected, tolerance):
    actual = result.cpu().double()
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    error = (actual - expected).abs()
    return {
        "max_abs_error": error.max().item(),
        "max_tolerance_ratio": (error / (tolerance + tolerance * expected.abs())).max().item(),
    }


def reference(torch, x, weight):
    values = x.cpu().double()
    return (
        values
        * torch.rsqrt(values.square().mean(dim=1, keepdim=True) + EPSILON)
        * (weight.cpu().double())
    )


def measure(torch, methods):
    for op in methods.values():
        for _ in range(5):
            op()
    torch.cuda.synchronize()
    samples = {name: [] for name in methods}
    names = list(methods)
    for rep in range(REPEATS):
        for name in names[:: 1 if rep % 2 == 0 else -1]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(BATCH):
                methods[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / BATCH)
    return samples


def run(output):
    import torch
    import triton
    from torch.profiler import ProfilerActivity, profile

    if not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA-enabled PyTorch environment with a visible NVIDIA GPU.")
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for dtype, tolerance in TOLERANCES.items():
        for rows, width, step in CASES:
            x, w = inputs(torch, rows, width, step, getattr(torch, dtype), "cuda")
            expected = reference(torch, x, w)
            methods = {"eager": partial(eager, torch, x, w), "fused": partial(fused, torch, x, w)}
            errors = {
                name: compare(torch, op(), expected, tolerance) for name, op in methods.items()
            }
            samples = measure(torch, methods)
            stats = {}
            for name, op in methods.items():
                filename = f"{dtype}-{rows}x{width}-s{step}-{name}.json"
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    op()
                    torch.cuda.synchronize()
                trace = output / filename
                prof.export_chrome_trace(str(trace))
                count = kernel_timeline(json.loads(trace.read_text())["traceEvents"])["kernels"]
                if name == "fused" and count != 1:
                    raise ValueError(f"{filename}: expected one fused kernel, found {count}")
                stats[name] = {
                    "samples_us": samples[name],
                    "median_us": median(samples[name]),
                    **errors[name],
                    "kernels": count,
                    "trace": filename,
                    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                }
            cases.append(
                {
                    "shape": [rows, width],
                    "dtype": dtype,
                    "strides": list(x.stride()),
                    "input_sha256": hashlib.sha256(
                        x.cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "weight_sha256": hashlib.sha256(w.cpu().numpy().tobytes()).hexdigest(),
                    "atol": tolerance,
                    "rtol": tolerance,
                    "methods": stats,
                }
            )
    result = {
        "experiment": "E02-RMSNorm-forward",
        "seed": SEED,
        "epsilon": EPSILON,
        "repeats": REPEATS,
        "calls_per_sample": BATCH,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "device_count": torch.cuda.device_count(),
        "compute_capability": torch.cuda.get_device_capability(),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("e02_kernel.py"))
        },
        "cases": cases,
    }
    pending = output / "summary.tmp"
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pending.rename(output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh results directory")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
