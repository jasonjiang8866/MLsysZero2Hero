"""E00 real PTX vector addition; no torch dependency and no CPU fallback.

Executed in a bounded subprocess by the CLI. Uses the pinned broker's CUDA
driver bindings and device/memory probe, then launches an actual arithmetic kernel.
"""

from __future__ import annotations

import ctypes as c
import hashlib
import json
import math
from pathlib import Path

from runpod_ci_broker.cuda_probe import Driver, GPUProbeError, probe

N = 257  # deliberately crosses a 128-thread block boundary
PTX = b""".version 7.0
.target sm_52
.address_size 64
.visible .entry vector_add(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u32 pn)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [pa];
    ld.param.u64 %rd2, [pb];
    ld.param.u64 %rd3, [pc];
    ld.param.u32 %r1, [pn];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE;
    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE:
    ret;
}
"""


def inputs() -> tuple[list[float], list[float]]:
    return ([float(i - 128) for i in range(N)], [float(i % 17 - 8) for i in range(N)])


def reference() -> list[float]:
    a, b = inputs()
    return [x + y for x, y in zip(a, b, strict=True)]


def check_output(values: list[float]) -> float:
    if len(values) != N or not all(math.isfinite(x) for x in values):
        raise GPUProbeError("invalid_kernel_output")
    error = max(abs(a - b) for a, b in zip(values, reference(), strict=True))
    if error != 0:  # small integer inputs have exactly representable FP32 sums
        raise GPUProbeError("kernel_numerical_mismatch")
    return error


def execute(output: Path, driver: Driver | None = None) -> dict:
    d = driver if driver is not None else Driver()
    observed = probe(d)
    if observed["driver_cuda_version"] < 12080:
        raise GPUProbeError("cuda_driver_below_12_8")
    ctx, module, function = c.c_void_p(), c.c_void_p(), c.c_void_p()
    d.call("cuDevicePrimaryCtxRetain", [c.POINTER(c.c_void_p), c.c_int], c.byref(ctx), 0)
    pointers = []
    try:
        d.call("cuCtxSetCurrent", [c.c_void_p], ctx)
        d.call(
            "cuModuleLoadData",
            [c.POINTER(c.c_void_p), c.c_void_p],
            c.byref(module),
            c.cast(c.c_char_p(PTX), c.c_void_p),
        )
        d.call(
            "cuModuleGetFunction",
            [c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p],
            c.byref(function),
            module,
            b"vector_add",
        )
        for _ in range(3):
            pointer = c.c_uint64()
            d.call("cuMemAlloc_v2", [c.POINTER(c.c_uint64), c.c_size_t], c.byref(pointer), N * 4)
            pointers.append(pointer)
        a, b = [(c.c_float * N)(*v) for v in inputs()]
        for pointer, host in zip(pointers[:2], (a, b), strict=True):
            d.call("cuMemcpyHtoD_v2", [c.c_uint64, c.c_void_p, c.c_size_t], pointer, host, N * 4)
        # Sentinel output catches skipped writes, including the final partial block.
        d.call("cuMemsetD32_v2", [c.c_uint64, c.c_uint, c.c_size_t], pointers[2], 0x7FC00000, N)
        count = c.c_uint(N)
        arguments = (c.c_void_p * 4)(*[c.cast(c.byref(p), c.c_void_p) for p in [*pointers, count]])
        d.call(
            "cuLaunchKernel",
            [c.c_void_p, *([c.c_uint] * 7), c.c_void_p, c.POINTER(c.c_void_p), c.c_void_p],
            function,
            (N + 127) // 128,
            1,
            1,
            128,
            1,
            1,
            0,
            None,
            arguments,
            None,
        )
        d.call("cuCtxSynchronize", [])
        result = (c.c_float * N)()
        d.call("cuMemcpyDtoH_v2", [c.c_void_p, c.c_uint64, c.c_size_t], result, pointers[2], N * 4)
        error = check_output(list(result))
    finally:
        # Context release remains attempted even if unloading/freeing raises.
        try:
            for pointer in pointers:
                d.call("cuMemFree_v2", [c.c_uint64], pointer)
        finally:
            try:
                if module.value:
                    d.call("cuModuleUnload", [c.c_void_p], module)
            finally:
                d.call("cuDevicePrimaryCtxRelease_v2", [c.c_int], 0)
    payload = json.dumps(list(result), allow_nan=False).encode()
    (output / "output.json").write_bytes(payload)
    return {
        "device": observed,
        "kernel": "ptx_vector_add_fp32_v1",
        "elements": N,
        "max_abs_error": error,
        "kernel_sha256": hashlib.sha256(PTX).hexdigest(),
        "input_sha256": hashlib.sha256(json.dumps(inputs()).encode()).hexdigest(),
        "timing": "not_a_performance_benchmark",
        "framework_runtime": "not_tested",
    }


if __name__ == "__main__":
    import sys

    try:
        root = Path(sys.argv[1])
        metrics = execute(root)
        (root / "metrics.json").write_text(json.dumps(metrics, allow_nan=False) + "\n")
    except (GPUProbeError, ValueError, OSError, IndexError):
        print('{"status":"failed","code":"cuda_smoke_failed"}')
        sys.exit(2)
