"""E01 single-GPU timing and starvation lab. Run in an existing CUDA PyTorch environment."""

import argparse
import json
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

CASES = ("baseline", "host_cpu", "io_wait", "small_launches")
STEPS = 16
SIZE = 262_144
CHUNKS = 32
DELAY = 0.003


def kernel_timeline(events):
    """Measure the union of CUDA kernel intervals in one isolated profile."""
    spans = sorted(
        (event["ts"], event["ts"] + event["dur"])
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "kernel" and event.get("dur", 0) > 0
    )
    if not spans:
        raise ValueError("Trace contains no CUDA kernels. Check CUDA profiler/CUPTI access.")
    first, end = spans[0]
    active = end - first
    gaps = []
    for start, stop in spans[1:]:
        if start > end:
            gaps.append(start - end)
        active += max(0, stop - max(end, start))
        end = max(stop, end)
    return {
        "kernels": len(spans),
        "span_us": end - first,
        "kernel_active_us": active,
        "inter_kernel_gap_us": sum(gaps),
        "longest_gap_us": max(gaps, default=0),
    }


def run(output):
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    if not torch.cuda.is_available():
        raise ValueError("E01 requires CUDA PyTorch and a visible GPU.")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    host = torch.arange(SIZE, dtype=torch.float32)
    x = host.to(device)
    y = torch.empty((STEPS, SIZE), device=device)
    expected = torch.stack([host + step for step in range(STEPS)])
    rows = list(y.unbind())
    parts = [row.chunk(CHUNKS) for row in rows]
    inputs = x.chunk(CHUNKS)

    def batch(case, annotated=False):
        for step, row in enumerate(rows):
            with record_function(case + "/step") if annotated else nullcontext():
                if case in ("host_cpu", "io_wait"):
                    with record_function(case + "/producer") if annotated else nullcontext():
                        if case == "io_wait":
                            time.sleep(DELAY)
                        else:
                            deadline = time.thread_time() + DELAY
                            while time.thread_time() < deadline:
                                pass
                if case == "small_launches":
                    for src, dst in zip(inputs, parts[step], strict=True):
                        torch.add(src, step, out=dst)
                else:
                    torch.add(x, step, out=row)

    def reset():
        y.fill_(float("nan"))
        torch.cuda.synchronize(device)

    def check():
        if not torch.equal(y.cpu(), expected):
            raise ValueError("E01 FP32 output differs from the CPU reference.")

    samples = {case: [] for case in CASES}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    for case in CASES:
        batch(case)
        check()
    # Rotate the order so each case runs in each position once.
    for trial in range(len(CASES)):
        for case in CASES[trial:] + CASES[:trial]:
            reset()
            cpu_begin = time.thread_time()
            begin = time.perf_counter()
            start.record()
            batch(case)
            end.record()
            submitted = time.perf_counter()
            cpu_submit_ms = (time.thread_time() - cpu_begin) * 1000
            end.synchronize()
            finished = time.perf_counter()
            samples[case].append(
                {
                    "host_submit_ms": (submitted - begin) * 1000,
                    "host_thread_cpu_ms": cpu_submit_ms,
                    "wall_ms": (finished - begin) * 1000,
                    "stream_span_ms": start.elapsed_time(end),
                }
            )
            check()

    results = {}
    for case in CASES:
        reset()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            batch(case, annotated=True)
            torch.cuda.synchronize(device)
        check()
        trace = output / f"{case}.json"
        prof.export_chrome_trace(str(trace))
        timeline = kernel_timeline(json.loads(trace.read_text())["traceEvents"])
        timings = samples[case]
        wall = statistics.median(s["wall_ms"] for s in timings)
        results[case] = {
            "samples": timings,
            "median_wall_ms": wall,
            "steps_per_second": STEPS * 1000 / wall,
            "trace": trace.name,
            **timeline,
        }
    report = {
        "experiment": "E01",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device),
        "steps": STEPS,
        "elements_per_step": SIZE,
        "dtype": str(x.dtype),
        "chunks": CHUNKS,
        "producer_delay_ms": DELAY * 1000,
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory for this run")
    args = parser.parse_args()
    try:
        run(args.output)
    except (ImportError, ValueError, FileExistsError) as exc:
        parser.exit(2, f"E01: {exc}\n")


if __name__ == "__main__":
    main()
