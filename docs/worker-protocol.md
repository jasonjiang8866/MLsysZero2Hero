# E00 worker and result transport / E00 worker 与结果传输

**E00 controller update / 控制器更新** · The worker foundation described below now has a manual launcher, remote budget ledger, artifact receipts and independent recovery. The first GPU run is pending. / 下方记录底座实现时的状态。手动启动、远程账本、产物回执和独立恢复现已接入，首次 GPU 验收待执行。[Current E00 runbook / 运行说明](../runpod/e00.md)

[Platform (EN)](lab-platform.md) · [实验底座（中文）](lab-platform.zh-CN.md)

## Components / 组件

| Component | Contract / 约定 |
| --- | --- |
| `worker/Dockerfile` | Python 3.12, locked non-editable dependencies, baked source SHA and OCI label. Git/uv stay in the build stage. / 固定依赖，源码 SHA 写入镜像，构建工具不进入运行层。 |
| `mlsys_lab.worker` | Fixed E00 PTX calculation; one GPU, 24 GB minimum, numeric driver capability >= 12.8. / 固定计算，无任意脚本入口。 |
| `mlsys_lab.transport` | Authenticated RunPod HTTPS proxy, exact identity, bounded reads, recomputed correctness/hashes, atomic local export. / 认证下载、身份及正确性复验、原子发布。 |
| `mlsys_lab.cleanup` | Known-Pod ownership verification, DELETE, then two direct-404 + complete-inventory absence pairs. / 已知 Pod 的所有权和删除复验原语。 |
| Build workflow | PR/main CPU image build and explicit refusal to pass without CUDA. No cloud secret. / 无云 key 的构建与无 GPU 拒绝测试。 |
| Publication workflow | Manual main-only, source checks after registry pull, public-access check, immutable digest artifact. No GPU rental. / 手动发布和固定 digest 回执，不租 GPU。 |

## Worker input / worker 输入

The future trusted controller supplies `MLS_JOB_JSON` conforming to
[`worker-runtime.schema.json`](../schemas/worker-runtime.schema.json)'s `Job` model,
and a random 256-bit hex result capability in `MLS_RESULT_TOKEN`. The worker reads
its Pod ID from provider-injected `RUNPOD_POD_ID` and its build revision from
`/opt/mlsys/SOURCE_SHA`. The job contains the attempt, full source SHA, immutable
image digest, expected GPU, compute limit (1–60 seconds) and collection allowance
(1–300 seconds). It has no arbitrary command, URL, mount or account credential.

未来可信控制器提供上述严格格式的 Job 和每次独立的随机结果令牌；Pod ID 取自
供应商环境，源码版本取自镜像内文件。Job 不允许任意命令、URL、挂载或账户凭据。
令牌只能下载本次实验结果，不能管理 RunPod 或 GitHub 账户。

Never put the account `RUNPOD_API_KEY` or GitHub token in a Pod creation request.
The worker discards a provider-injected Pod-scoped key unused, removes the result
token from its environment and launches the CUDA child with a small environment
allowlist. A matching baked source SHA is a build check; it does not independently
attest an arbitrary image. The controller must bind the registry digest/source
evidence to the provider's actual Pod allocation before accepting results.

账户 key 留在控制器。worker 丢弃供应商自动注入的 Pod 范围 key，并以环境白名单
启动 CUDA 子进程。镜像内 SHA 检查不替代独立验证：控制器仍须将 registry digest、
源码证据和实际 Pod 分配对应起来。

## Transport / 传输

Only exact GET paths are served on port 8000 behind the RunPod TLS proxy:

- `/v1/status`: authenticated phase and runtime fingerprint.
- `/v1/manifest`: runtime identity plus the E00 bundle manifest.
- `/v1/files/metrics.json` and `/v1/files/output.json`: immutable payloads.

只支持以上路径；没有 shell、任意文件或管理接口。请求日志不输出令牌或请求原文。
SHA-256 runtime 指纹包含 Job 和 Pod ID，防止串到其他 Pod、旧 attempt 或不同源码。

The server starts before compute. A failed computation exposes `failed`, with no
successful result export. Proxy 404/429/502/503/504 responses during readiness may
be retried inside the overall deadline; 401/403 are terminal. File fetches do not
silently restart. The collector caps status at 4 KiB, the envelope at 64 KiB and each
payload at its declared size (maximum 64 KiB). It rechecks selected GPU, all hashes
and every output value before Linux `renameat2(RENAME_NOREPLACE)` publishes the
staged directory atomically. A competing destination is never overwritten.

先启动服务，再执行计算。ready 前的代理未就绪可限时重试，认证错误立即失败。
下载失败或数据篡改不发布结果。文件长度、数值、哈希及 GPU 都复验后，Linux 原子
操作才让完整目录可见；不覆盖旧产物或并发下载目录。

Once a trusted runtime record exists, collection can run on a Linux controller:

```bash
# MLS_RESULT_TOKEN must already be supplied privately by the controller.
uv run --locked python -m mlsys_lab.transport \
  --runtime out/runtime.json --output out/e00-retrieved --timeout-seconds 180
```

`manifest.json` remains the E00 numerical contract; `transfer.json` preserves the
runtime envelope. Local export is **not** durable artifact upload. Acceptance still
requires a verified GitHub artifact receipt, budget reconciliation and verified
Pod deletion. Neither worker process exit nor server TTL deletes a billable Pod.

本地取回不等于持久上传；仍须有 GitHub artifact 回执、预算结算和 Pod 删除证据。
进程退出或服务超时不等于停止计费，必须由外部控制器负责删除。

## Cleanup boundary / 清理边界

`cleanup_owned()` is deliberately not exposed as an unauthenticated CLI taking an
arbitrary Pod ID. It requires a known allocation and trusted journal callbacks.
Before DELETE it re-reads ownership and durably checkpoints intent. It checks
repository/campaign/attempt/run markers, Pod name and ID, image, GPU count/model,
cloud, data center and absence of persistent mounts. A changed owner, failed
checkpoint or mismatched allocation prevents deletion. An ambiguous DELETE is
reconciled by GET before another deletion request. A 204 alone is never success.

清理函数必须由可信账本提供 allocation；不能输入任意 Pod ID 就执行删除。删除前
再次核对所有权并持久记录意图。账本写入失败或身份不同就拒绝删除。DELETE 超时先
查询状态，不盲目重复。直接 404 与完整库存两次均确认不存在，才返回删除成功。

The controller persists cleanup intent and evidence on broker-state. A separate workflow reconciles the completed owner attempt. / 控制器在 broker-state 持久化清理意图及证据，独立恢复任务对原尝试重新核验。

## Validation and next gate / 验证与下一门槛

Tests include real local HTTP requests around fake-driver results, corrupted and
oversized downloads, wrong identities, authentication refusal, readiness deadlines,
atomic publication races, mismatched cleanup ownership, lost DELETE acknowledgments,
incomplete inventories and failed durable checkpoints. Image CI checks build and
CPU refusal. Real PTX JIT/GPU execution and the RunPod TLS path still need E00.

测试覆盖实际本地 HTTP，但计算结果来自明确标记的 driver 替身。真实 PTX JIT、GPU
运算及 RunPod TLS 路径仍须通过 E00 验证。E00 子上限为 US$1 / 200 GPU 分钟，计入项目累计额度。

The transport and cleanup design reuse patterns and APIs from the pinned
[runpod-ci-broker](https://github.com/jasonjiang8866/runpod-ci-broker/tree/5822d7f79051f8b4945a6c394743fcd1772d8405),
MIT, Copyright (c) 2026 JYC. Its implementation remains an installed dependency;
its separate paid campaign and spending history are not imported.
