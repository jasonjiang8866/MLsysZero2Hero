# MLsysZero2Hero

[简体中文](README.zh-CN.md)

A bilingual, experiment-driven path from machine learning development to **ML Systems / AI Infrastructure and GPU Kernel Engineering**.

For learners who can already code and train a model. “Zero” means no prior systems-optimization experience. Python, PyTorch tensors/autograd, basic linear algebra, Linux and Git are prerequisites; C++/CUDA foundations are taught on the kernel track.

**Status: curriculum planning. No runnable lessons or paid experiment workflow yet.** This repository does not claim that the planned labs or benchmarks have been completed.

## Start with a question

**Why would 2,048 GPUs spend time waiting?**

Our opening case follows DeepSeek-V3's H800 training system: GPU idle time, expert routing, intra-node/inter-node communication, and computation–communication overlap. Later chapters revisit these questions through small experiments and a detailed case study. R1 extends the story to reinforcement-learning post-training on V3-Base; it is not the source of all V3 infrastructure techniques.

The opening asks questions rather than requiring knowledge of DualPipe or MoE. See the [case-study plan and sources](docs/deepseek-case-study.md).

## Learning route

| Stage | Scope | Evidence learners produce |
| --- | --- | --- |
| Common core | Measurement, profiling, GPU execution and memory, basic kernels | Reproducible diagnosis and numerical tests |
| Kernel engineering | CUDA/Triton, GEMM, reductions, efficient attention, tuning | Correct kernel and shape-dependent benchmark |
| ML systems | Collectives, topology, distributed training/attention, MoE, serving | Scaling/latency report and a justified design |
| Integration | DeepSeek case study, RL systems and capstone | End-to-end analysis, code and reproducibility record |

Both tracks learn communication and profiling fundamentals. Advanced implementation depth differs by track.

- [Full curriculum / 课程总表](docs/curriculum.md)
- [Development milestones / 开发里程碑](docs/roadmap.md)
- [RunPod experiment plan / 实验与预算](runpod/README.md)

## How the course is built

Each lesson will have separate English and Chinese explanations with the same lesson ID, shared code/tests/data/figures, bilingual terminology, prerequisites, hardware requirements, exercises and solutions. Every experiment states a hypothesis, baseline, correctness gate, measurement method, observations and limitations. A well-explained optimization that fails to improve performance is a valid result.

Experiments are tiered: CPU/reference-trace access; one GPU; 2–4 GPUs; verified multi-node hardware. Simulated topology results will never be presented as physical cluster measurements. Single-GPU learners can complete the core.

## Initial experiment envelope

The owner authorized **US$20 and 1,000 minutes** for this project's initial experiments on 2026-11-16 (Asia/Singapore). We conservatively account time as aggregate allocated GPU-minutes, including startup and retries. The plan keeps all charges within US$20, including temporary storage and cleanup. These are planning limits; enforcement is not implemented yet.

The owner reports a RunPod repository secret is configured. Its name, availability and API access remain to be validated safely in Actions. No credential should be pasted into an issue, log or artifact.

## License

[MIT](LICENSE). External papers, figures and datasets retain their respective licenses; course diagrams and explanations should be original with attribution.
