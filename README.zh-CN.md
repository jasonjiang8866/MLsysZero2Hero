# MLsysZero2Hero

[English](README.md)

面向已有机器学习和编程基础的学习者，以实验和项目为核心，走向 **ML Systems / AI Infra 与 GPU Kernel Engineering**。

“Zero”指零系统优化经验。先修要求：Python、PyTorch tensor/autograd、基本线性代数、Linux 和 Git；kernel 路线补充 C++/CUDA 基础。

**当前状态：M1 实验底座开发中。** 已实现有时限的 CUDA kernel smoke 命令、结果验证、预算预留及只读 RunPod 查询。CPU 测试不代表真实 GPU 验收，付费生命周期接入和完整课程仍待完成。见[实验底座](docs/lab-platform.zh-CN.md)。

## 从一个问题开始

**2,048 张 GPU，为什么还需要等待？**

开篇以 DeepSeek-V3 的 H800 训练系统引出 GPU 空转、expert 路由、节点内外通信，以及计算与通信重叠。后续章节用小实验逐步回答，最后完整复盘。R1 用于延伸 V3-Base 上的强化学习后训练；V3 基础设施技术不统一归到 R1 名下。

[开始 L00 · 跟着一个 token 走过 H800 集群](lessons/l00/README.zh-CN.md)。用 15 分钟预测三种专家布局的跨节点发送量，再检验算子快一倍能省几毫秒。只需 Python 标准库。

## 学习路线

| 阶段 | 范围 | 学员交付 |
| --- | --- | --- |
| 共同基础 | 测量、profiling、GPU 执行与内存、基础算子 | 可复现诊断和数值测试 |
| Kernel engineering | CUDA/Triton、GEMM、reduction、高效 attention、调优 | 正确算子与不同 shape 的性能分析 |
| ML systems | 集合通信、拓扑、分布式训练/attention、MoE、serving | 扩展效率和延迟报告、设计依据 |
| 综合应用 | DeepSeek 案例、RL 系统与毕业项目 | 端到端分析、代码和复现记录 |

两条路线都学习通信和 profiling 基础，进阶实现深度不同。

- [完整课表](docs/curriculum.md)
- [开发里程碑](docs/roadmap.md)
- [RunPod 实验与预算](runpod/README.md)

## 课程标准

每课未来提供独立中英文讲义，相同 lesson ID，共用代码、测试、数据和图表；包括术语对照、先修知识、硬件要求、练习和参考答案。每次实验包含假设、基线、正确性门槛、测量方法、观察与局限。合理解释一次未能加速的优化也是有效成果。

实验分级：CPU/参考 trace、单卡、2–4 卡、已验证网络条件的多节点。模拟拓扑结果不能冒充真实集群测量，单卡学员可以完成核心课程。

## 首轮实验额度

项目所有者于 2026-11-16（新加坡时间）授权 **US$20、1000 分钟**。暂按所有已分配 GPU 的累计分钟数保守记账，包含启动和重试；计划将临时存储、清理等费用也纳入 US$20 内。累计准入逻辑已实现并通过离线测试，[E00 控制器](runpod/e00.md)现已接入课程远程账本。

用户表示已配置 RunPod 仓库 secret；具体名称、可用性和 API 连通性尚待 Actions 安全预检验证。不要将凭据写入 issue、日志或产物。

## 许可证

[MIT](LICENSE)。引用的外部论文、图表和数据保留各自许可证；课程解释和示意图应自行制作并注明来源。
