"""One Triton program computes one row of weighted, forward-only RMSNorm."""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X, W, Y, n: tl.constexpr, row_stride, col_stride, eps: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < n
    x = tl.load(X + row * row_stride + col * col_stride, mask=valid, other=0).to(tl.float32)
    w = tl.load(W + col, mask=valid, other=0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / n + eps)
    tl.store(Y + row * n + col, x * inv_rms * w, mask=valid)
