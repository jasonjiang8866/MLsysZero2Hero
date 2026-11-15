# DeepSeek case study / 贯穿式案例

[Curriculum / 课表](curriculum.md)

Status: teaching design, not a reproduction of DeepSeek training. / 状态：教学设计，不是 DeepSeek 训练复现。

## Opening: 10–15 minutes / 开场：10–15 分钟

**Why would 2,048 GPUs spend time waiting? / 2,048 张 GPU，为什么还需要等待？**

Use three original teaching visuals: a node/network topology, a deliberately simplified compute/communication timeline, and token-to-expert routing. Label every illustrative timeline as a schematic, not a measured DeepSeek trace. / 制作节点拓扑、简化计算通信时间线、token 路由三张原创示意图，明确不是 DeepSeek 实测 trace。

Ask learners:
1. Why can a faster GPU fail to make training faster? / 为什么 GPU 更快，训练却未必更快？
2. What happens when a token needs experts on different nodes? / expert 跨节点后数据怎么走？
3. When can communication be overlapped, and what resources does it consume? / 何时能重叠，重叠占用什么资源？
4. Can changing routing or placement reduce traffic? / 路由和放置如何影响通信量？
5. What new scheduling problems arise during RL rollouts? / RL 生成阶段新增什么调度问题？

Give the questions and a course map at the beginning; defer kernels and scheduling derivations until L07–L14. / 开场建立问题地图，具体实现留到后半程。

## Evidence and attribution / 证据与归属

| Reported fact / 报告事实 | Primary source / 来源 | Teaching use / 教学用途 |
| --- | --- | --- |
| V3 trained on 2,048 H800 GPUs, 8 per node, NVLink/NVSwitch within nodes and IB between nodes / 集群配置 | V3 §3.1 | Explain communication hierarchy / 解释层级 |
| DualPipe overlaps computation/communication and reduces bubbles / 重叠与气泡 | V3 §3.2.1 | Study dependency and scheduling constraints / 调度与依赖 |
| Node-limited routing and custom cross-node All-to-All / 节点受限路由及通信 kernel | V3 §2.1.2, §3.2.2 | Co-design routing and communication / 路由通信协同 |
| Communication uses 20 of H800's 132 SMs / 通信占用 SM | V3 §3.5.1 | Overlap has a resource cost / 重叠有资源代价 |
| R1 and R1-Zero start from V3-Base / R1 基于 V3-Base | R1 repository and report | Separate base-model infrastructure from RL post-training / 区分基座基础设施与后训练 |

These are author-reported facts, not independently reproduced results in this repository. Do not infer unspecified rack wiring from node descriptions, describe bandwidth limits as eliminated, attribute Ring Attention to V3 without evidence, or quote V3 training costs as R1's complete development costs. / 以上是作者报告，非本仓库复验；不推断未披露机架布线，不声称带宽限制消失，不无依据归属 Ring Attention，不混用 V3 与 R1 总成本。

## Later deep dive / 后期深入讲解

L14 revisits DualPipe's schedule and memory tradeoffs, routing/traffic constraints, dispatch/combine kernels, balancing and FP8 accuracy/performance. Learners perform small counterfactuals: faster compute with unchanged network; skewed expert demand; weaker overlap; a different node layout. / 最后复盘调度与显存代价、通信、均衡、FP8，分析换卡不换网、路由偏斜、重叠不足和节点布局变化。

The low-budget capstone reproduces selected mechanisms at small scale or uses explicitly labeled models. It does not promise H800-scale reproduction. / 小预算实验仅验证选定机制或明确标注的模型，不承诺复现 H800 集群规模。

## Sources / 来源

- [DeepSeek-V3 Technical Report v2](https://arxiv.org/html/2412.19437v2)
- [DeepSeek-R1 original report v1](https://arxiv.org/abs/2501.12948v1)
- [DeepSeek-R1 official repository](https://github.com/deepseek-ai/DeepSeek-R1)

Record exact source revisions in completed lessons. / 正式课程记录具体引用版本。
