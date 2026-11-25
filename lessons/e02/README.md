# E02 · Write a fused RMSNorm kernel

[简体中文](README.zh-CN.md) · [Course map](../../docs/curriculum.md) · [E01 trace evidence](../../docs/results/e01-45288496471.md)

In E01, splitting the same addition into 512 launches left more time between kernels. Can a single launch normalize and scale a row faster than eager PyTorch? Implement the forward pass, then find a shape where kernel count alone is a poor predictor of latency.

## Write the kernel

Start with a 2-D tensor of activations and one learned weight per column. For each row, compute `y = x * rsqrt(sum(x²) / width + 1e-5) * weight`. Accumulate in FP32 and store in the input dtype. For a row [3, 4] and weights [2, 1], the expected result is approximately [1.6971, 1.1314]. This is an inference exercise covering the forward pass.

1. Open [the kernel](../../src/mlsys_lab/e02_kernel.py). Keep its signature, cover the function body, and implement it yourself. Assign one program to each row and generate a power-of-two block of column offsets.
2. Use the input's row and column strides for addressing. For width 1,000 the block has 1,024 lanes. Determine what the 24 padded lanes should contribute and which divisor to use. Mask weight loads and output stores too.
3. Predict what happens with an all-zero row, a row of −512 in FP16, and a view taking every other column. Check your implementation against the [worked kernel](../../src/mlsys_lab/e02_kernel.py) and [numerical tests](../../tests/test_e02.py).

## Check correctness, then measure

From the repository root, use Python 3.12 on Linux with CUDA-enabled PyTorch 2.9.0, Triton 3.5.0, one NVIDIA GPU and working CUPTI profiling. The runner uses the current device. Choose a fresh output directory each time.

```bash
git rev-parse HEAD
python -m torch.utils.collect_env
PYTHONPATH=src python -m mlsys_lab.e02 --output out/e02-rmsnorm-01
```

For a numerical check on a CPU-only Linux machine, install PyTorch 2.9.0 (CPU), Triton 3.5.0, NumPy and pytest. Run `TRITON_INTERPRET=1 PYTHONPATH=src python -m pytest -q tests/test_e02.py`. This executes the same kernel in the interpreter. Timing and launch counts require CUDA.

The runner compares both implementations with a CPU FP64 reference on identical inputs. It exercises FP32 and FP16, row counts 1/64/512 at width 1,024, and 64 rows at width 1,000 with column stride 1 or 2. For each case it warms up both paths, alternates their timing order, and saves seven CUDA-event samples of 20 calls each. Each trace profiles a separate single call. The summary appears after every case passes.

## Interpret and submit

- Open `summary.json`. For each matched case report maximum absolute error, both kernel counts, median latency in µs, and the ratio eager/fused. Retain the raw samples and the traces alongside the GPU environment and source commit.
- At 64 rows, compare widths 1,024 and 1,000. Then hold width at 1,000 and switch only the input stride. Compare the eager and fused timelines in Perfetto. Which intermediate arrays does fusion avoid writing?
- **Worked reasoning.** Inactive lanes contribute zero to the sum, and the divisor is the logical width. FP32 accumulation avoids overflow when squaring −512. A stride of two can spread neighboring loads across more memory transactions. A single row launches only one Triton program, so most SMs may have no work. Test those predictions with the measurements. A slower fused result is a valid finding.

This first E02 exercise covers uncompiled eager PyTorch and a hand-written forward kernel. Softmax, backward kernels and E02 hardware acceptance remain open.
