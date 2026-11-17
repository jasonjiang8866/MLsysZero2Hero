# Development plan / 开发计划

[Home](../README.md) · [Curriculum](curriculum.md) · [Experiments](../runpod/README.md)

Planning date: 2026-11-16 Asia/Singapore. Initial inspected main: 3c459c851a216a3716b6e6b5225f76e93e9a8cde (LICENSE only); no open PRs or Actions runs at inspection. / 初始 main 仅 LICENSE，无开放 PR 或 Actions 运行记录。

The first release is a coherent single-GPU learning path, not a claim that the entire curriculum is ready. / 首版交付完整单卡学习路径，全课表分阶段开发。

| Milestone / 里程碑 | Deliverables / 交付 | Exit gate / 验收 | Dependency and status / 依赖与状态 |
| --- | --- | --- | --- |
| M0: Course foundation / 课程骨架 | Bilingual entry, L00 case plan, full syllabus, budget/runbook / 双语入口、案例、课表、实验计划 | Links, coverage, budget sums and scope reviewed / 链接、覆盖、预算检查 | Merged in PR #1 / PR #1 已合并 |
| M1: Reproducible lab platform / 实验底座 | Locked environment, result schema, CPU CI, RunPod quote/preflight controller, cleanup and ledger tests / 环境锁、结果格式、预检、清理、记账 | Offline tests pass; safe secret-presence check; no paid push/PR triggers; then E00 smoke / 离线通过、安全预检后 smoke | M0; foundation implemented, live activation pending / 底座已实现、真实激活待完成 |
| M2: First learning slice / 首个学习闭环 | L00–L04: bilingual lessons, measurement, idle diagnosis, first kernel exercises and solutions / 开场到算子首个闭环 | E01/E02 artifacts, traces, shape/dtype correctness and explained results / 实测及正确性证据 | M1; planned |
| M3: Attention on one GPU / 单卡 Attention | L05–L06, dense/blockwise/optimized comparisons / 分类、分块与优化 | E03 output/gradient checks, memory report; optional E06 small serving extension / 输出梯度与显存证据 | M2; planned |
| M4: Communication and distributed mechanisms / 通信与分布式机制 | L07/L09 foundations, L10 ring prototype, L11 MoE imbalance / 通信、ring、MoE | E04/E05 only on verified topology; small-scale correctness, scaling/imbalance report / 核实拓扑后实测 | M3; hardware/budget conditional |
| M5: Cluster and serving depth / 集群与服务深化 | L08/L12/L13, cross-rack networking, serving/RL/reliability case material / 跨机架、服务、RL、可靠性 | Sources and traces attributable; simulations labeled; real T3 runs separately costed / 来源可追溯、模拟明确 | M4; full T3 execution not promised in initial budget |
| M6: Integration / 综合验收 | L14 DeepSeek deep dive and capstone rubric / 案例复盘与毕业项目 | Reproduce selected mechanism and explain end-to-end tradeoffs / 验证机制、解释取舍 | Relevant modules complete; planned |

## M1 implementation status / M1 实现状态

See the [English platform guide](lab-platform.md) and [中文实验底座](lab-platform.zh-CN.md). The foundation implements package/lockfile, CPU CI, PTX kernel smoke, bundle verification, cumulative CAS budget reservations and read-only preflight. The [worker/transport increment](worker-protocol.md) adds a container, authenticated collection and ownership-checked cleanup primitives. Paid creation, image publication, remote ledger activation, upload receipts and independent recovery wiring remain pending. / 已增加 worker 镜像构建、认证传输和所有权清理原语；付费创建、镜像发布、远程账本、上传回执及独立恢复接线仍待完成。

## M1 acceptance contract / M1 验收约定

M1 should contain:
- Project packaging and a pinned environment, CPU-only correctness tests and a GPU smoke command. / 项目环境、CPU 测试、GPU smoke。
- A result schema with source SHA, image digest, dependency versions, GPU/driver/topology, input hashes, seeds, numerical errors, timing method, costs and cleanup evidence. / 完整结果记录。
- A manual-only RunPod workflow and inspected controller; read-only quote/inventory preflight before creation. / 手动运行、先报价和库存。
- Atomic cumulative budget reservations, unknown-creation reconciliation, timeouts, artifact recovery, finally/always cleanup and independently verified deletion. / 累计预算预留、防重复创建、超时、产物回收与删除复验。
- Tests for caps, retries, cancellation, missing secrets, incompatible CUDA, ambiguous creation and cleanup failure. / 上限、重试、取消与失败路径测试。

After those gates, run E00 within its sub-cap using the existing budget authorization. Do not ask for the same budget again. Merge permission and execution-route availability are tracked separately; this planning request does not imply merging future implementation PRs. / 门槛通过后沿用已有预算执行 E00；不重复申请同一预算，合并与运行通道单独记录。

## Release quality / 发布质量

Both languages must describe the same behavior. Code, tests and numerical data are shared. Preserve failed experiments and negative results. Each measured result includes immutable provenance and retrieved artifacts. CPU tests cannot certify GPU correctness or performance. / 双语行为一致、共用实现、保留失败与负结果、产物可追溯；CPU 测试不代表 GPU 已验收。

Plan progress is separate from experiment status: planned, preflight-blocked, running, completed, accepted/rejected, cleanup-verified. / 课程进度与实验生命周期分别记录。
