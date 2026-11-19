# RunPod experiments and budget / 实验与预算

[Roadmap / 路线图](../docs/roadmap.md)

## Current state / 当前状态

M1 foundation is implemented: pinned dependencies, local CUDA kernel smoke, result verification, cumulative budget logic and a manual GET-only preflight workflow. The [E00 workflow](e00.md) integrates manual launch, durable receipts and independent recovery. Live acceptance is pending. / E00 已接入手动启动、持久回执和独立清理，真实 GPU 验收待运行。

The owner reports RUNPOD_API_KEY is configured; presence/connectivity remain unverified for this repository until its manual main-only preflight runs. No key is retrieved or printed. No RunPod resources were created by this change. / key 是否存在及连通性待本仓库预检确认，本次未创建 RunPod 资源。

The current connector exposes no workflow-dispatch tool; local gh is unauthenticated. No browser session was probed. Broker source and its successful regression evidence were inspected; its campaign-bound paid controller cannot be used as this project's budget authority. / 已检查 Broker 源码与成功回归，但不能借用其项目预算与付费控制器。

## Authorization record / 授权记录

- Project: jasonjiang8866/MLsysZero2Hero; initial course validation experiments.
- User authorization dated 2026-11-16 Asia/Singapore: US$20 and 1,000 minutes. / 首轮课程实验额度。
- Conservative planning interpretation: US$20 all-in and 1,000 aggregate allocated GPU-minutes across runs/retries, whichever binds first. A two-GPU Pod for 30 minutes uses 60 GPU-minutes. Include billed startup, idle and cleanup periods. / 更保守地按总费用和累计 GPU 分钟控制，双卡 30 分钟记 60 分钟，含计费启动、空闲与清理时间。
- The cap does not reset with a retry, PR, workflow run or conversation. No automatic replenishment. / 重试、PR、运行或新对话都不重置。
- No particular GPU model has yet been selected. Proposed default: one RTX 4090 for fitting workloads; alternatives require workload/compatibility/quote checks. Specific model selection and its authorization scope must be resolved before launch; existing budgets for other repositories are unrelated. / 提议单卡 4090 优先，具体型号及授权范围启动前核实，不借用其他项目预算。
- No persistent volume creation/deletion or unrelated resource modification is included. Prefer ephemeral synthetic inputs. / 不包含持久卷增删及无关资源修改。
- This change incurs US$0 in RunPod charges. Account/project resources and unrelated charges remain unverified. / 本次未触发 RunPod 费用，账户现有资源及其他支出未验证。

## Allocation / 实验额度

On 2026-11-19 the owner raised the cumulative E00 time allowance to 200 GPU-minutes. The additional 170 minutes come from the time reserve. Existing reservations count toward the new limit. / 11 月 19 日批准 E00 时间上限调至 200 分钟，从时间预备额度调入 170 分钟，历史预留继续累计。

These are **maximum envelopes, not price quotes or spending targets**. Reductions are expected when experiments finish early. Full course implementation may require later reprioritization or new funding; the present envelope funds initial validation only. / 以下是上限，不是报价或必须花完的目标；仅用于首轮验证。

| ID | Experiment / 实验 | Tier | USD ceiling | Aggregate GPU-minutes |
| --- | --- | --- | ---: | ---: |
| E00 | Provision → CUDA smoke → artifact → deletion / 创建到回收闭环 | T1 | 1 | 200 |
| E01 | Timing and GPU starvation / 计时与空转诊断 | T1 | 3 | 150 |
| E02 | RMSNorm/softmax and fusion / 基础算子与融合 | T1 | 3 | 180 |
| E03 | Dense vs blockwise attention / Attention 正确性与显存 | T1 | 3 | 160 |
| E04 | Collectives and minimal Ring Attention / 集合通信与最小 ring | T2 conditional | 4 | 160 |
| E05 | MoE skew and balance / MoE 偏斜与均衡 | T1/T2 conditional | 2 | 80 |
| E06 | Small serving workload / 小模型 serving | T1 conditional | 2 | 40 |
| Reserve | Recovery, storage, transfer, cleanup / 恢复、存储、传输、清理 | — | 2 | 30 |
| **Total** | | | **20** | **1000** |

Normal workloads total US$18 / 970 GPU-minutes. Keep US$2 / 30 GPU-minutes for recovery. Retries debit the same project ledger. Do not automatically consume the recovery reserve for extra benchmarks. / 正常实验 18 美元/970 分钟，先保留 2 美元/30 分钟恢复空间。

T2 requires a verified multi-GPU host/topology; two unrelated marketplace Pods do not establish a usable RDMA/NVLink cluster. MIG slices do not automatically substitute for physical multi-GPU communication labs. Cross-rack performance is not an acceptance promise of this budget. / 多卡需核实同机/网络能力，不能将两个随意租的 Pod 当成 RDMA 集群，MIG 也不自动等于多卡实验。

## Controller design gates / 控制器开发门槛

1. Manual workflow_dispatch only for paid work; CPU CI may run on PRs. Validate workload enum, source SHA, image digest, GPU count, cost and time caps before creation. / 付费仅手动触发，启动前校验所有参数。
2. Keep RUNPOD_API_KEY in Actions Secrets; never send it into the workload Pod or artifacts. Check only presence and sanitized API results. / key 留在 Actions，Pod 不携带账户 key。
3. Query verified API catalog, quote, region, storage and existing project-owned Pods. Record quote time and total Pod/hour price. Compare host CUDA numerically using >= required minimum, separately validate driver/image/GPU compatibility. / 当前报价与库存、数值版 CUDA 最低要求及兼容性。
4. Persist a ledger across runs with an atomic reservation/lock and one active experiment. Record charges already incurred, pending reservations and unresolved billable resources. Block new work when state is missing/ambiguous. / 跨运行持久记账、原子预留、单实验并发，状态不明就阻止新开。
5. Derive maximum wall time from remaining money after storage/transfer/cleanup reserve, total Pod hourly rate, and remaining aggregate GPU-minutes divided by GPU count. Use the stricter per-experiment cap too. Include startup time and billing granularity. / 费用、GPU 分钟和单实验上限取最严格者。
6. GitHub timeout alone does not cap Pod billing. Add a workload deadline plus an independent ownership-scoped reaper; test controller crash/cancellation. These safeguards do not create a provider-enforced dollar hard cap. / Actions 超时不等于停止计费，需要独立清理机制；不夸称供应商硬性金额封顶。
7. Label each resource by repository/run/experiment. If creation response is lost, reconcile inventory before retrying. Never delete unrelated Pods or persistent volumes. / 资源明确归属，创建结果不明先查库存，禁止盲重试。
8. Recover results, hash artifacts, run finally/always cleanup, then independently query inventory to verify deletion. A successful delete request or scheduled reaper is not proof. / 先回收、再删除、再独立复查。
9. Before dispatch discover the actual tool route: connector, already-authenticated CLI, then authorized browser. Do not bypass an explicit denial or use paid push triggers to work around missing dispatch. / 发现实际可用通道，不绕过拒绝或借 push 偷启动。
10. A ready launch record must show provenance, live quote, GPU scope, both remaining caps, acceptance criteria and cleanup route. Reuse the existing valid budget authorization. / 启动记录具体可审阅，沿用已有有效预算。

## Acceptance and evidence / 验收与证据

E00: real CUDA kernel executes; results downloaded and checksums verified; project-owned temporary Pod deletion independently confirmed. E01: injected CPU/I/O/launch bottlenecks explained with traces. E02/E03: reference numerical and gradient checks where applicable plus repeated measurements; no required speedup. E04: collective data integrity, ring output/gradient checks and recorded physical topology. E05: expert token counts, max/mean load, dropped-token counts, numerical output and slowest-rank time. E06: documented model/input identity, cache state, TTFT/TPOT/P99 and throughput under a fixed workload. / 每项有具体数值、性能或生命周期证据，不以进程退出零代替验收。

For every run record: experiment ID, source SHA, workflow URL/ref, immutable image digest, inputs/hashes, seed, GPU/driver/dependencies/topology, timestamps and GPU-minutes, reserved/estimated/measured cost, numerical tolerances/errors, raw metrics and trace paths, artifact checksums, owned Pod identity, deletion evidence, remaining budget and unresolved issues. Never include credentials or raw Pod environments. / 结果记录必须可复验且不含凭据。

No paid experiment is marked accepted until its artifacts are retrieved and cleanup is verified. / 产物未回收或清理未复验，不能标记最终验收完成。
