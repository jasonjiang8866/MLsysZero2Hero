# M1 lab platform

**E00 controller update / 控制器更新** · The worker foundation described below now has a manual launcher, remote budget ledger, artifact receipts and independent recovery. The first GPU run is pending. / 下方记录底座实现时的状态。手动启动、远程账本、产物回执和独立恢复现已接入，首次 GPU 验收待执行。[Current E00 runbook / 运行说明](../runpod/e00.md)

[简体中文](lab-platform.zh-CN.md) · [Roadmap](roadmap.md)

## Implemented

- Python 3.12 package, uv lockfile, pinned Broker revision, CPU CI and wheel build.
- E00 CUDA Driver API smoke: memory roundtrip followed by a real FP32 vector-add
  kernel over 257 elements. The final partial block is checked. No CPU fallback.
- A 1–60 second child-process limit; account credentials are excluded from the child.
- An E00 bundle contract: declared source revision and optional image digest,
  input/kernel hashes, observed GPU/driver facts, raw numerical output, file sizes
  and SHA-256 hashes. Verification recomputes correctness and rejects stale identities,
  missing/modified files, symlinks, oversized files and nonfinite output.
- Cumulative project/experiment budget reservations using integer micro-USD and
  aggregate GPU-seconds. CAS writes prevent competing reservations; ambiguous writes
  require exact readback. Existing attempts and unresolved resources block new work.
- A manual main-only RunPod **GET-only** catalog/inventory workflow, using the
  owner's repository secret only on its controller step.

- A source-bound course worker, authenticated result server, bounded HTTPS collector,
  Linux atomic result publication, and ownership-checked cleanup primitives.
- Docker build/CPU-refusal CI and a manual main-only GHCR publication workflow.
  See the [worker protocol and operational boundaries](worker-protocol.md).

**M1 is in progress, not fully accepted.** CPU tests use explicit driver/provider
doubles; they do not demonstrate real GPU execution. PTX JIT compatibility and actual
kernel execution remain to be checked on a GPU. Use the manual E00 workflow for a cloud run.

## Local commands

```bash
uv sync --locked --python 3.12
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build
```

On an existing compatible single-GPU Linux machine with a driver supporting CUDA
12.8 or newer, run the following from a clean checkout. This command does not rent
a GPU, modify driver configuration or delete a Pod.

```bash
uv run --locked mlsys-lab smoke --source-sha "$(git rev-parse HEAD)" --output out/e00
uv run --locked mlsys-lab verify --source-sha "$(git rev-parse HEAD)" --output out/e00
```

Use a new output directory per attempt. Failure exits 2 and never writes a success
manifest. A timeout terminates the child; stopping a process is **not** proof of
cloud-resource deletion. The course container checks its baked build revision and publication checks the
registry image label. The live controller must still verify the selected immutable
image and provider allocation; the standalone smoke command binds results to declared
provenance and does not attest the source of an arbitrary binary.

The driver-reported CUDA version is host capability, not a verified toolkit or
PyTorch runtime. E00 does not verify PyTorch, Triton, throughput or training quality.
Its output explicitly states that this is not a performance benchmark and that cloud
acceptance has not been evaluated.

## Read-only RunPod preflight

After this workflow exists on main, manually run
[Course RunPod read-only preflight](https://github.com/jasonjiang8866/MLsysZero2Hero/actions/workflows/runpod-preflight.yml).
There are no paid inputs or creation methods. The workflow reports secret presence,
authentication/query failures, candidate prices, regional capacity, minimum-CUDA
checks and project-scope inventory. An empty catalog does not imply bad credentials.
Successful GET queries do not imply a GPU was rented or the course smoke passed.

Discovery inspects 4090, 3090, 5090 and PRO 6000 MIG 24 GB as candidates, with 4090
first. This candidate list is not a paid hardware-selection authorization. The
USD 3/GPU-hour discovery filter is a planning filter, not an observed quote or a new
spending allowance. Storage/transfer quotes must be checked before paid admission.

The task's pinned Broker memory-smoke image identifies a known reference runtime
for discovery; it is not the unpublished course kernel worker. No account secrets,
raw Pod environments or provider response bodies are included in outputs.

## Budget persistence and limits

The authorization remains **US$20 / 1,000 aggregate GPU-minutes**. Normal workloads
may reserve only US$18 / 800 GPU-minutes, preserving recovery headroom. Each E00–E06
sub-cap from the course plan also applies cumulatively across retries.

`Budget` uses the pinned Broker's Store protocol: `SQLiteStore` is for local offline
tests, while `GitHubContentsStore` can persist CAS records on a dedicated branch.
The remote course ledger was initialized and read back on broker-state. Never use
an ephemeral runner SQLite database as the production budget ledger. Missing state
fails closed; bootstrap is explicit and cannot overwrite an existing ledger.

The accounting model retains reservations after cleanup; greater observed usage
supersedes the reserve for subsequent admissions. It records overruns instead of
discarding evidence that exceeds a limit. `close()` requires an evidence hash, but
the E00 controller verifies the underlying receipt. A hash supplied by
a caller does not itself prove deletion. These are software admission limits, not
provider-enforced spend guarantees.

## Why the existing Broker paid workflow is not reused directly

Broker revision `5822d7f79051f8b4945a6c394743fcd1772d8405` contains a working CUDA
memory smoke, but its paid controller is bound to its own repository, campaign IDs,
predecessor records and US$20/5-hour history. This repository reuses the GET adapter,
strict models, driver bindings and storage primitives without copying that spending
history or weakening those guards.

## Live E00 acceptance

Publish the current worker, verify the ancillary price bounds and run the manual workflow. See the [E00 runbook](../runpod/e00.md).

## Sources and licensing

- [Broker pinned source](https://github.com/jasonjiang8866/runpod-ci-broker/tree/5822d7f79051f8b4945a6c394743fcd1772d8405)
- [Broker regression evidence](https://github.com/jasonjiang8866/runpod-ci-broker/blob/5822d7f79051f8b4945a6c394743fcd1772d8405/docs/gpu-regression-postmortem.md)
- [CUDA kernel launch API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html)
- [PTX ISA 7.0](https://docs.nvidia.com/cuda/archive/11.0/parallel-thread-execution/index.html)

Broker dependency and driver-double test pattern are MIT licensed; attribution:
Copyright (c) 2026 JYC, runpod-ci-broker. Its license is preserved in that dependency.
