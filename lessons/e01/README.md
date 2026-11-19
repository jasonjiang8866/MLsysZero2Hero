# E01 · Why is the GPU waiting?

[简体中文](README.zh-CN.md) · [Curriculum](../../docs/curriculum.md)

You own a training loop whose throughput has dropped. A dashboard shows low GPU utilization. What measurement would tell you whether to work on the input pipeline or the kernels? This L01–L02 lab gives you two host delays that can produce similar GPU timelines, plus a launch-bound workload with exactly the same arithmetic.

## Run the experiment

Use a Linux machine with one visible NVIDIA GPU, CUDA-enabled PyTorch and CUDA profiler (CUPTI) support. From the repository root in that Python environment, run:

```bash
python -m torch.utils.collect_env > e01-environment.txt
PYTHONPATH=src python -m mlsys_lab.e01 --output out/e01-01
```

Choose a fresh output directory each time. The command uses your existing device. On completion, look for four Chrome trace JSON files and `summary.json`. A missing summary signals an incomplete run. To study a peer’s run on CPU, open each trace in Perfetto and read the summary alongside it. Record the repository commit with `git rev-parse HEAD`.

## Predict before measuring

Every case adds the step index to the same 262,144 FP32 values for 16 steps. Inputs and outputs already reside on the GPU. Predict which metric changes most in each row.

| Case | One changed variable | Evidence to look for |
| --- | --- | --- |
| baseline | One vector add per step | A short host dispatch followed by a device kernel |
| host_cpu | The producer consumes 3 ms of thread CPU time per step | A producer range before dispatch and a gap between device kernels |
| io_wait | The producer sleeps for 3 ms per step | Similar gaps while the producer thread waits. This simulates a blocked input reader |
| small_launches | Split each add across 32 disjoint slices | More kernel launches for the same number of output elements |

The runner warms up each case, rotates case order across four repetitions and synchronizes before each trial. It verifies every output row against a CPU reference after timing. Trace collection is a separate pass with host annotations.

## Read the evidence

1. Compare all four wall-time samples and the median steps/s. Compare host_thread_cpu_ms for host_cpu and io_wait. A sleeping thread can accumulate wall time with little thread CPU time. Host submission time ends after enqueue. Wall time includes waiting for completion. The CUDA event span can include idle intervals while the host prepares work.
2. Open one pair of traces. Expand the host thread and the GPU stream. Zoom to two consecutive steps, identify their launch calls and follow the associated kernels. Locate the producer range before the next launch.
3. Report the kernel count, longest inter-kernel gap and kernel coverage (`kernel_active_us / span_us`). Coverage uses the union of kernel intervals between the first start and the last end. Other processes, copy-engine activity and time before the first kernel are outside that metric. Instrumentation changes launch timing, so use the unprofiled trials for throughput.

**Exercise.** A synthetic trace has kernels at 0–20 and 100–120 µs. Compute coverage. Can this trace distinguish a CPU transform from blocked I/O? How would you check whether fragmentation is more important than input readiness?

**Worked answer.** Coverage is 40/120, about 33%. The 80 µs gap identifies when this stream has no observed kernel. Correlate the next launch with producer ranges and thread CPU time or scheduler/I/O evidence to identify the cause. For fragmentation, compare kernel counts and wall time at fixed work against the baseline, then batch adjacent operations and repeat. A short enqueue timer alone cannot demonstrate a speedup.

**Submit** the environment and commit, four raw samples per case, two annotated trace excerpts and one proposed intervention. Identify an alternative cause the traces cannot rule out. Full E01 hardware acceptance requires a real run with numerically correct outputs and a diagnosis supported by its traces.

[PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) · [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
