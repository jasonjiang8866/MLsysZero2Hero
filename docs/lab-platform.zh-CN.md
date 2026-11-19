# M1 实验底座

**E00 controller update / 控制器更新** · The worker foundation described below now has a manual launcher, remote budget ledger, artifact receipts and independent recovery. The first GPU run is pending. / 下方记录底座实现时的状态。手动启动、远程账本、产物回执和独立恢复现已接入，首次 GPU 验收待执行。[Current E00 runbook / 运行说明](../runpod/e00.md)

[English](lab-platform.md) · [开发路线](roadmap.md)

## 已实现

- Python 3.12 项目、uv 锁文件、固定 Broker 提交、CPU CI 和 wheel 构建。
- E00 CUDA Driver API smoke：显存读写后执行真实 FP32 向量加法 kernel，
  检查 257 个元素和最后一个不完整 block，无 CPU fallback。
- 子进程运行上限 1–60 秒，子进程环境不携带账户凭据。
- E00 结果格式：声明的源码提交、可选镜像 digest、输入/kernel 哈希、GPU/driver
  观测值、原始输出、文件大小及 SHA-256。独立验证数值和哈希，拒绝版本不匹配、
  缺失/篡改文件、符号链接、超大文件及非有限数值。
- 项目与单实验累计预算账本：整数微美元、累计 GPU 秒、CAS 并发保护、写入结果
  不明时精确读回；重复实验和未解决的资源阻止下一次预留。
- 仅在 main 手动运行的 RunPod **GET-only** 报价/库存预检工作流，账户 key
  只提供给控制器的查询步骤。

- 课程 worker、源码版本检查、带认证的结果服务、有界 HTTPS 下载、Linux 原子
  结果发布及检查所有权的清理原语。
- Docker 构建/CPU 拒绝测试 CI，以及仅在 main 手动发布 GHCR 的工作流。
  详见[worker 协议和运行边界（双语）](worker-protocol.md)。

**M1 尚在进行，未整体验收。** CPU 测试使用明确标记的 driver/provider 替身，
不代表真实 GPU 执行。PTX 的 JIT 兼容性与真实 kernel 运行仍待 GPU 验证。
真实实验通过 E00 手动工作流启动。

## 本地使用

```bash
uv sync --locked --python 3.12
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build
```

在已有的单卡 Linux 机器上，driver 需支持 CUDA 12.8 或更高。在干净 checkout
中运行以下命令；它不会租 GPU、修改驱动配置或删除 Pod。

```bash
uv run --locked mlsys-lab smoke --source-sha "$(git rev-parse HEAD)" --output out/e00
uv run --locked mlsys-lab verify --source-sha "$(git rev-parse HEAD)" --output out/e00
```

每次使用新输出目录。失败返回退出码 2，不能生成成功 manifest。超时终止子进程，
但进程退出不等于云资源已删除。课程容器检查烘焙的源码版本，发布工作流验证
镜像源码 label；真实控制器仍须核对固定 digest 和供应商分配。单独的 smoke 命令
只绑定声明的版本信息，不证明任意二进制的真实构建来源。

driver 报告的 CUDA 版本是宿主机能力，不是已经验证的 toolkit 或 PyTorch 版本。
E00 不验证 PyTorch、Triton、吞吐或训练质量；结果明确标注非性能 benchmark，
且尚未进行云端生命周期验收。

## RunPod 只读预检

工作流进入 main 后，手动运行
[Course RunPod read-only preflight](https://github.com/jasonjiang8866/MLsysZero2Hero/actions/workflows/runpod-preflight.yml)。
没有付费参数和创建接口。它记录 secret 是否存在、认证/查询错误、候选报价、
区域库存、最低 CUDA 检查和项目范围内库存。库存为空不等于 key 错误；GET 查询
成功也不代表租卡或课程 smoke 成功。

候选包括 4090、3090、5090、PRO 6000 MIG 24 GB，4090 优先。
候选列表不等于具体付费选卡授权。3 美元/GPU 小时是查询过滤条件，不是当前报价
或新增预算。启动前仍须核实存储与传输费用。

查询 task 中固定的 Broker 镜像仅标识已知的显存 smoke 参考环境，不是尚未发布的
课程 kernel 镜像。本次不输出账户 key、原始 Pod 环境或供应商原始响应。

## 预算与持久化

沿用 **US$20／1000 累计 GPU 分钟**。普通实验最多预留 18 美元／970 分钟，
保留恢复空间；原 E00–E06 单项上限也跨重试累计。

`Budget` 使用固定 Broker 的 Store 协议：`SQLiteStore` 用于离线本地测试；
`GitHubContentsStore` 可在独立分支持久化 CAS 记录。课程远程账本已在 broker-state 初始化并读回验证。禁止将临时 runner 上的 SQLite 当作生产账本。状态缺失即拒绝，
初始化必须明确执行且不能覆盖已有账本。

清理后仍保留原预留额；观测用量更大时，下一次准入按更大的值计算。超额事实会
保留，不因超过上限而丢弃。`close()` 要求证据哈希，但真实控制器必须验证对应
回执，调用者提供一个哈希本身不能证明已删除。这是软件准入控制，不是供应商
强制的美元封顶保证。

## 为什么不直接调用 Broker 的付费工作流

Broker 固定提交 `5822d7f79051f8b4945a6c394743fcd1772d8405` 已有真实显存
smoke 证据，但付费控制器绑定自身仓库、固定 campaign、前序记录及 20 美元/
5 小时历史。本项目复用 GET adapter、严格数据模型、driver 绑定和存储原语，
不复制那段预算历史，也不放松它的检查。

## E00 云端验收

发布当前课程镜像，核实磁盘等费用上限，再手动运行 E00。参见[运行说明](../runpod/e00.md)。

## 来源和许可

- [Broker 固定源码](https://github.com/jasonjiang8866/runpod-ci-broker/tree/5822d7f79051f8b4945a6c394743fcd1772d8405)
- [Broker 回归记录](https://github.com/jasonjiang8866/runpod-ci-broker/blob/5822d7f79051f8b4945a6c394743fcd1772d8405/docs/gpu-regression-postmortem.md)
- [CUDA kernel 启动 API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html)
- [PTX ISA 7.0](https://docs.nvidia.com/cuda/archive/11.0/parallel-thread-execution/index.html)

Broker 依赖及 driver 测试替身模式使用 MIT 许可，归属 Copyright (c) 2026 JYC,
runpod-ci-broker。原依赖保留其许可证。
