# Curriculum / 课程总表

[Home](../README.md) · [中文首页](../README.zh-CN.md)

Status: all lesson implementations are **planned / 尚未实现**. This bilingual planning index is not a substitute for separate English/Chinese lesson pages.

Hardware tiers: **T0** CPU, original simulations or sourced traces; **T1** one compatible GPU; **T2** 2–4 GPUs with topology recorded; **T3** multiple nodes with verified network capabilities. Hardware examples are requirements, not a promise of RunPod inventory.

| ID | Module / 模块 | Scope / 内容 | Lab and acceptance / 实验与验收 | Prerequisite / 先修 |
| --- | --- | --- | --- | --- |
| L00 | Why systems matter / 系统工程的价值 | DeepSeek-V3/H800 opening; R1 context / 开场案例与问题地图 | T0: map bottlenecks to course modules / 将瓶颈映射到后续课程 | ML + coding |
| L01 | Measurement / 正确测量 | Async execution, CUDA events, warmup, repetitions, percentiles / 异步、计时、预热、重复、分位数 | T1: expose a false speedup; distinguish host dispatch from completed GPU work / 拆解虚假加速 | L00 |
| L02 | GPU idleness / GPU 空转 | DataLoader, CPU/I/O starvation, launch gaps, synchronization, stragglers / 数据不足、启动间隙、同步与慢任务 | T1: inject one bottleneck at a time; trace-backed diagnosis / 逐一注入瓶颈并用 trace 证明 | L01 |
| L03 | GPU architecture / GPU 架构 | SM/warp, registers, shared memory, HBM, Tensor Cores, coalescing, Roofline / 执行与存储层级、算术强度 | T1: compare measured bandwidth and arithmetic intensity / 带宽与计算瓶颈分析 | L01 |
| L04 | Kernel foundations / 算子基础 | C++/CUDA primer, Triton, reduction, RMSNorm, softmax, GEMM, fusion, tuning / 基础编程、融合、调优 | T1: correctness across shapes/dtypes, including tails and strides; benchmark / 数值、尾块、步长与性能 | L02–03 |
| L05 | Attention taxonomy / Attention 分类 | SDPA, causal masks, MHA/MQA/GQA, MLA, local/sparse/linear attention / 语义、参数共享与近似的区别 | T0/T1: calculate compute and KV storage; compare only equivalent semantics / 计算与缓存分析、等价语义对照 | L03–04 |
| L06 | Efficient attention / 高效 Attention | Online softmax, tiling, FlashAttention forward/backward, prefill/decode, PagedAttention / 分块、反向与缓存分页 | T1: dense reference vs blockwise forward/gradients; peak memory / 前向梯度误差及峰值显存 | L05 |
| L07 | GPU communication / 卡间通信 | PCIe, NVLink/NVSwitch, P2P, NUMA; NCCL collectives and point-to-point / 拓扑、集合通信、点对点 | T2: latency/bandwidth vs message size, rank placement; scaling analysis / 消息大小与放置实验 | L01–03 |
| L08 | Cluster networking / 跨节点与机架 | NIC, IB/RoCE, RDMA, GPUDirect RDMA, leaf-spine, oversubscription, congestion, ECN/PFC / 网络路径与拥塞 | T0 model + T3 measurements when available; keep evidence classes separate / 模型分析与真实测量分别标注 | L07 |
| L09 | Distributed training / 分布式训练 | DDP, FSDP/ZeRO, TP, PP, SP/CP, microbatches, recomputation, pipeline bubbles / 并行策略、重计算与气泡 | T2/T3: match effective batch and workload; measure memory and scaling / 固定工作量与有效 batch 对照 | L07; L08 for T3 |
| L10 | Distributed attention / 分布式 Attention | Ring Attention, Ulysses, context parallelism, causal imbalance, overlap / 序列分片、因果负载与重叠 | T2: minimal ring implementation; outputs and gradients vs dense reference, then overlap / 先正确再重叠 | L06–07; L09 |
| L11 | MoE systems / MoE 系统 | Top-k router, capacity, dispatch/combine, EP, All-to-All, grouped GEMM, balancing / 路由、容量、通信与均衡 | T1 routing model; T2 distributed experts: skew vs balanced loads, dropped tokens, slowest rank / 偏斜负载对照 | L04, L07, L09 |
| L12 | Inference systems / 推理系统 | Continuous batching, KV/prefix cache, quantization, speculative decoding, prefill/decode separation / 批处理、缓存与调度 | T1: fixed workload/SLO, TTFT/TPOT/P99, throughput, memory and quality / 延迟约束下的吞吐与质量 | L05–07 |
| L13 | Operations and RL systems / 集群运行与 RL 系统 | Placement, checkpoints, recovery, observability, cost; rollout, variable sequence lengths, weight sync / 调度恢复、生成与权重同步 | T0/T1 fault scenarios and rollout trace; distinguish educational model from R1 reproduction / 故障案例与轨迹分析 | L09, L11–12 |
| L14 | DeepSeek deep dive and capstone / 案例复盘与毕业项目 | DualPipe, node-limited routing, communication kernels, FP8 tradeoffs; return to L00 / 协同设计与适用边界 | Explain the original case and defend one reproducible optimization / 复盘案例并提交可复现优化 | Relevant preceding modules |

## Concepts that must remain distinct / 必须区分的概念

- GPU utilization measures activity, not achieved peak compute. Diagnose empty timelines separately from inefficient active kernels. / 利用率不等于有效算力；区分无工作和低效执行。
- FlashAttention optimizes exact-attention execution and memory traffic; Ring Attention distributes attention across devices; PagedAttention concerns paged KV access/management. / 三者分属不同层面，可组合，不能作为同类算法简单排名。
- Training router balance, runtime expert scheduling and inference expert placement/replication solve different problems. / 训练均衡、运行时调度和推理放置分别评估。
- Ring Attention does not remove dense attention's quadratic arithmetic. Cross-node does not prove cross-rack. / 分布式不消除密集 attention 的二次计算量，跨节点不等于跨机架。
- Fewer communication bytes, faster links, better overlap and improved load balance are separate levers. / 减少通信量、提高带宽、重叠和均衡分别做消融。

## Lesson contract / 每课交付规范

Separate English/Chinese pages; shared code and tests; aligned IDs and terminology; prerequisite check; learning outcomes; hardware/cost tier; hypothesis and baseline; correctness tolerances justified per dtype/workload; runnable exercise; solution; measured report or explicit not-run label; limitations; primary references. / 独立双语讲义、共用代码测试、学习目标、硬件成本、假设基线、按精度设定容差、练习解答、实测或未运行标识、局限和一手资料。

No universal speedup target. Compare equivalent precision, shapes, effective batch, caching state and workloads; report regressions as well as wins. / 不设通用“必须加速”门槛，比较条件必须一致并报告退化。

## Primary references / 一手资料

- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA GPU utilization definition](https://nvidia.custhelp.com/app/answers/detail/a_id/3751)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [Ring Attention paper](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses](https://www.deepspeed.ai/tutorials/ds-sequence/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [Megatron Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [vLLM benchmarking](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [DeepSeek case sources](deepseek-case-study.md)

Pin tool versions and source revisions when implementing each lesson; links to latest docs are discovery references, not environment pins. / 实现时固定依赖与来源版本，latest 链接不是运行环境版本锁。
