# E02 · 动手写一个融合的 RMSNorm 算子

[English](README.md) · [课程总表](../../docs/curriculum.md) · [E01 真机 trace](../../docs/results/e01-45288496471.md)

E01 把相同的加法拆成 512 次启动，GPU 在 kernel 之间多等了一段时间。这次反过来做：把一行数据的归一化和缩放合进一次启动。先写出正确的前向算子，再找出仅凭 kernel 数量很难预测延迟的 shape。

## 实现 kernel

输入是二维激活张量，每一列有一个权重。逐行计算 `y = x * rsqrt(sum(x²) / width + 1e-5) * weight`。用 FP32 累加，输出转回输入 dtype。若一行输入为 [3, 4]，权重为 [2, 1]，结果应约为 [1.6971, 1.1314]。本节练习推理前向计算。

1. 打开 [kernel 文件](../../src/mlsys_lab/e02_kernel.py)，保留签名，遮住函数体自己实现一遍。让每个 Triton program 处理一行，把列偏移扩展到二的整数次幂。
2. 用输入的行、列 stride 寻址。宽度为 1,000 时，计算块有 1,024 个位置。多出的 24 个位置应该为求和贡献多少？分母取多少？权重读取和输出写回也要加 mask。
3. 预测全零行、全为 −512 的 FP16 行，以及隔列取值的张量会怎样。再对照[参考实现](../../src/mlsys_lab/e02_kernel.py)和[数值测试](../../tests/test_e02.py)。

## 先验数值，再测性能

在仓库根目录运行。环境为 Linux、Python 3.12、带 CUDA 的 PyTorch 2.9.0、Triton 3.5.0、一张 NVIDIA GPU，以及可用的 CUPTI profiler。程序使用当前设备，每次选择新的输出目录。

```bash
git rev-parse HEAD
python -m torch.utils.collect_env
PYTHONPATH=src python -m mlsys_lab.e02 --output out/e02-rmsnorm-01
```

仅有 Linux CPU 时，安装 PyTorch 2.9.0 CPU 版、Triton 3.5.0、NumPy 和 pytest，执行 `TRITON_INTERPRET=1 PYTHONPATH=src python -m pytest -q tests/test_e02.py`。解释器会执行同一份 kernel 来核对数值。延迟与启动次数需要在 CUDA 上采集。

两种实现使用相同输入，分别与 CPU FP64 参考值比较。矩阵覆盖 FP32 和 FP16：宽度 1,024 时分别取 1、64、512 行，宽度 1,000 时固定 64 行并比较列步长 1 和 2。两种实现各自预热，交替改变计时顺序，记录七组 CUDA event 样本，每组 20 次调用。另行采集单次调用的 trace。所有组合通过后才生成汇总。

## 分析并提交

- 读 `summary.json`，为每组输入记录最大绝对误差、两种实现的 kernel 数、延迟中位数（µs）及 eager/fused 耗时比。保留原始样本、trace、GPU 环境与源码提交号。
- 固定 64 行，先对比宽度 1,024 和 1,000，再固定宽度 1,000，只改变列步长。在 Perfetto 中对照 eager 与 fused 时间线。哪些中间张量省去了一次写回？
- **参考思路。** 填充位置为求和贡献零，分母取实际列数。先转 FP32 可以避免 −512 的平方在 FP16 中溢出。隔列取数可能需要更多显存事务。只输入一行时，融合版只启动一个 Triton program，多数 SM 可能没有工作可做。用实测检验这些预测，融合后变慢同样值得报告。

本节覆盖 PyTorch eager 与手写的前向 kernel。Softmax、反向算子和 E02 真机验收仍待后续完成。
