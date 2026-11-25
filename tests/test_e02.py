"""Use TRITON_INTERPRET=1 with torch/triton installed for CPU kernel validation."""

import os

import pytest

from mlsys_lab import e02


@pytest.mark.parametrize("dtype", e02.TOLERANCES)
@pytest.mark.parametrize("rows,width,step", e02.CASES)
def test_rmsnorm_matrix(rows, width, step, dtype):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if not interpret and not torch.cuda.is_available():
        pytest.skip("Use TRITON_INTERPRET=1 or a CUDA device")
    device = "cpu" if interpret else "cuda"
    x, weight = e02.inputs(torch, rows, width, step, getattr(torch, dtype), device)
    assert x.stride() == (width * step, step)
    if rows > 1:
        x[0].zero_()
        x[1].fill_(1e-4)
        x[2].fill_(-512)  # Squaring in FP16 would overflow.
    expected = e02.reference(torch, x, weight)
    for impl in (e02.eager, e02.fused):
        output = impl(torch, x, weight)
        assert output.dtype == x.dtype and output.shape == x.shape
        e02.compare(torch, output, expected, e02.TOLERANCES[dtype])
