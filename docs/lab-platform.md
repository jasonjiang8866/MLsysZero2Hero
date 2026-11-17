# M1 lab platform

[简体中文](lab-platform.zh-CN.md) · [Roadmap](roadmap.md)

## Implemented in this change

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

**M1 is in progress, not fully accepted.** CPU tests use explicit driver/provider
doubles; they do not demonstrate real GPU execution. PTX JIT compatibility and actual
kernel execution remain to be checked on a GPU. No paid controller is enabled here.

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
cloud-resource deletion. Later container integration must supply and independently
verify the build revision and immutable image digest. The current command binds
results to declared provenance; it does not attest the source of an arbitrary binary.

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
The remote course ledger has not been bootstrapped or integration-tested. Never use
an ephemeral runner SQLite database as the production budget ledger. Missing state
fails closed; bootstrap is explicit and cannot overwrite an existing ledger.

The accounting model retains reservations after cleanup; greater observed usage
supersedes the reserve for subsequent admissions. It records overruns instead of
discarding evidence that exceeds a limit. `close()` requires an evidence hash, but
the future live controller must verify the underlying receipt. A hash supplied by
a caller does not itself prove deletion. These are software admission limits, not
provider-enforced spend guarantees.

## Why the existing Broker paid workflow is not reused directly

Broker revision `5822d7f79051f8b4945a6c394743fcd1772d8405` contains a working CUDA
memory smoke, but its paid controller is bound to its own repository, campaign IDs,
predecessor records and US$20/5-hour history. This repository reuses the GET adapter,
strict models, driver bindings and storage primitives without copying that spending
history or weakening those guards.

## Remaining M1 activation gates

1. Build/publish a course worker at an immutable digest with a verified source label.
   Add an authenticated readiness/result server around this E00 implementation.
2. Integrate a course-specific lifecycle controller with a durable remote budget
   ledger, execution identity and one create intent per reserved attempt.
3. Add tested artifact retrieval/upload receipts, ownership-checked termination,
   direct absent-resource checks and independent inventory verification.
4. Add independent cancellation/crash recovery and cleanup-only reaping. Verify
   recovery ownership and interrupted-run takeover before exposing paid dispatch.
5. Obtain fresh quote/inventory, select hardware within existing authorized scope,
   verify image pull/driver compatibility, then run E00 within its US$1/30-minute
   allocation and the remaining project budget.

The existing budget authorization persists. Merge/release scope and a usable
workflow-dispatch route remain separate from that budget. At this implementation
check, no dispatch tool was exposed and local gh was unauthenticated; no browser
session was probed. This is capability state, not an automatic approval rejection.

## Sources and licensing

- [Broker pinned source](https://github.com/jasonjiang8866/runpod-ci-broker/tree/5822d7f79051f8b4945a6c394743fcd1772d8405)
- [Broker regression evidence](https://github.com/jasonjiang8866/runpod-ci-broker/blob/5822d7f79051f8b4945a6c394743fcd1772d8405/docs/gpu-regression-postmortem.md)
- [CUDA kernel launch API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html)
- [PTX ISA 7.0](https://docs.nvidia.com/cuda/archive/11.0/parallel-thread-execution/index.html)

Broker dependency and driver-double test pattern are MIT licensed; attribution:
Copyright (c) 2026 JYC, runpod-ci-broker. Its license is preserved in that dependency.
