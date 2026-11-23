# E01 · GPU 在等什么？

[English](README.md) · [课程表](../../docs/curriculum.md)

[查看 RTX 4090 原始 trace 与实测解读](../../docs/results/e01-45288496471.md)

训练吞吐下降，监控面板显示 GPU 利用率偏低。你会先改数据管线，还是先合并小算子？这节 L01–L02 实验让你亲手制造两种外观相似的供数延迟，再用相同的计算量观察过多 kernel launch 的影响。

## 运行

准备一台 Linux 主机、一张可见的 NVIDIA GPU，以及支持 CUDA 的 PyTorch 和 CUDA profiler（CUPTI）。在该 Python 环境中，从仓库根目录执行：

```bash
python -m torch.utils.collect_env > e01-environment.txt
PYTHONPATH=src python -m mlsys_lab.e01 --output out/e01-01
```

每次使用新的输出目录。命令直接使用已有 GPU。完整运行会生成四份 Chrome trace 和一份 `summary.json`；缺少汇总说明运行未完成。只有 CPU 的同学可以用 Perfetto 打开同伴的 trace，对照汇总完成读图练习。用 `git rev-parse HEAD` 记录代码版本。

## 先预测，再测量

每组都在同样的 262,144 个 FP32 元素上做 16 步加法，每一步加上步号。输入、输出提前放在 GPU 上。先预测每组中变化最大的指标。

| 组别 | 改变的条件 | 时间线上的观察点 |
| --- | --- | --- |
| baseline | 每步一次向量加法 | 主机提交后出现对应的 GPU kernel |
| host_cpu | 生产线程每步消耗 3 ms 线程 CPU 时间 | 下一次提交前有 producer 区间，相邻 kernel 之间出现间隙 |
| io_wait | 生产线程每步休眠 3 ms，模拟等待读取 | 线程等待期间也会出现相似间隙；这里没有测量真实磁盘 |
| small_launches | 把每步加法切成 32 个互不重叠的切片 | 输出元素数相同，每步提交更多小 kernel |

每组先预热，接着轮换顺序测四次。每轮开始前同步 GPU，结束计时后用 CPU 参考值核对每一步的全部输出。程序另跑一轮采集 trace，并给主机操作添加标注。

## 怎样读结果

1. 对照四次原始 wall time 和按中位数计算的 steps/s。对照 host_cpu 和 io_wait 的 host_thread_cpu_ms；线程休眠时，墙钟时间仍在走，而线程 CPU 时间几乎不增加。 主机提交时间在工作入队后截止，wall time 一直量到 GPU 完成。两个 CUDA event 之间的跨度也可能包含主机供数造成的空档。
2. 选两组 trace，展开主机线程和 GPU stream。放大相邻两步，找到 launch 调用和它们对应的 kernel，再看下一次提交之前的 producer 区间。
3. 比较 kernel 数、最长间隙和覆盖率（`kernel_active_us / span_us`）。分子是 kernel 区间并集，分母从首个 kernel 开始到末个结束。该指标只描述这次采集的 kernel，无法说明其他进程、复制引擎或首个 kernel 之前的活动。Profiler 会扰动短算子的时序，吞吐取自单独的无 profiler 测量。

**练习。** 一段模拟 trace 只包含 0–20 和 100–120 µs 两个 kernel。覆盖率是多少？你能仅凭这条轨道分辨 CPU 预处理和阻塞 I/O 吗？怎样验证小算子过多是否拖慢吞吐？

**参考答案。** 覆盖率为 40/120，约 33%。80 µs 的间隙说明这条 stream 在此期间没有观测到 kernel。把下一次 launch 与 producer 区间对齐，再结合线程 CPU 时间、调度或 I/O 证据判断原因。检查小算子时，固定工作量，对照基线的 kernel 数和同步后的 wall time，然后合并相邻操作重测。只看到提交时间变短不足以证明加速。

**提交物。** 环境和提交版本、每组四次原始计时、两张带标注的时间线局部图，以及一项优化建议。指出一个现有证据仍无法排除的原因。E01 真机验收需要保存数值正确的实测结果，并根据时间线完成归因。

[PyTorch CUDA 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html) · [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
