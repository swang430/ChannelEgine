# 导入 Meta-3D 聚簇轨迹并重建频响

`mimo_ota_simulator.imported_cluster_trajectory` 接收
`schemaVersion: "meta3d-cluster-trajectory/1"`。这是 **Meta 调用未修改的
Lauraycs 已有接口后组织的交换格式，不是 Lauraycs 原生文件**。
旧的 `lauraycs-cluster-trajectory/1` 声明会被拒绝，不能据此声称支持原生导入。

此模块迁移自 Meta 的 `scripts/s5_evaluation/cluster_import.py`。
Meta 后续应直接调用此库，不再保留校验和计算算法副本；HTTP 采集和 Meta
入口切换由集成任务负责。本模块不发 HTTP 请求，也不修改 Lauraycs。

## 公开调用

安装 ChannelEgine 后，Meta 可直接使用以下签名（与迁移前一致）：

```python
from mimo_ota_simulator.imported_cluster_trajectory import (
    read_trajectory, validate_trajectory, sampled_responses,
)

# 文件必须是 Meta 组织的上述交换格式。
data = read_trajectory("trajectory.json")
# 已取得字典时可直接校验；返回原对象，不删除未知字段。
validate_trajectory(data)
# 相对载频的明确频率偏移，单位 Hz；返回顺序与该列表一致。
rows = sampled_responses(data, [0.0, 125e6])
for row in rows:
    if row["status"] == "unavailable":
        print(row["timeSeconds"], row["reason"])
    else:
        print(row["timeSeconds"], row["real"], row["imag"])
```

`validate_trajectory(data)` 保留完整数据，不复制或变换输入。调用者若要修改
数据，应自行复制。`read_trajectory(path)` 最多读取 256 MiB，拒绝重复 JSON
字段和非有限数值。格式、计数、单位、时间和必需字段不符合约定时拒绝导入。
文件读取错误与 JSON 解码错误向调用方传播。

## 数据含义与限制

- 时间轴必须非负且严格递增；保留原始非均匀时刻及全部帧，不补点、不插值。
- 复数 H 的实部、虚部原样保留。频响仅按每帧全部簇计算
  `Σ H × exp(-j × 2π × Δf × delayNs × 1e-9)`；不归一化、不重新聚簇。
- `sampled_responses(data, offsets_hz)` 要求明确、非空、有限数值列表；
  输出是已有时刻的复标量模型频响，标记 `model_reconstruction`，不是测量值，
  不代表完整天线矩阵、硬件合成结果或正式性能判定。
- `no_paths` 与 `no_valid_paths` 都保留在时间轴中，响应为
  `status="unavailable"`、`real=None`、`imag=None`，原因保留原状态，不能当零使用。
- 来源的 `contentVersion`、`softwareVersion`、`additionalLoss` 必须显式存在，
  未知时允许 `None`；角度坐标系、传播类型和多普勒的允许未知项同样不补造。
- `clusterIdentity` 必须为 `frame-local-only`：簇编号只在帧内有意义，
  相邻帧相同编号不证明同一条物理传播路径。本库不推断跨帧身份或多普勒演化。
- 结构校验不能证明任务、MPDB 内容或软件版本真实，也不能证明设备适用性。
  本库不输出 ASC、不操作硬件、不调用其他合成器。

## 本地验证

在 ChannelEgine 仓库根目录运行：

```bash
.venv/bin/python -m pytest tests/test_imported_cluster_trajectory.py -q --basetemp=.pytest_tmp
.venv/bin/python scripts/check_imported_cluster_mutations.py
```

变异脚本只修改已加载模块的内存代码，逐项执行测试，最后恢复内存实现。
每个替换锚点必须唯一命中；只有测试失败才算检出，收集错误不算检出。
这些是软件测试，不能替代真实接口采集或仪器验证。

## 按方向交付全帧采样结果

在仓库根目录可直接运行，集成服务也可在进程内调用同一函数。

```bash
python -m mimo_ota_simulator.sampled_trajectory trajectory.json --direction downlink --offsets-hz 0 5000000
```

`synthesize_sampled_trajectory(data, offsets_hz, *, direction)` 每次接收一个方向。
调用者依据资产来源指定 downlink 或 uplink。如果输入显式声明方向，两者必须一致。
返回对象的 schemaVersion 为 `channelengine-sampled-trajectory/1`，direction 记录方向，
offsetsHz 保存有序频偏。frames 包括每个样本的 sampleIndex、frameId、internalLinkId、
timeSeconds、status、reason、real、imag。最后一项对应输入末帧，包括末帧缺测的情况。
sampling 为 `source_samples_only`，帧间没有输出。仅适用于选定收发天线对的模型复频响；
坐标变换、探头映射、连续快衰落、频带内物理精度和仪器回放均需各自的证据。
质量表缺少有效路径计数时，filteredPaths 显式填 null。簇数仍须在输入路径数和预算以内。

### 2026-11-19 实际导出数据的软件复核

输入来自同一次仿真的上下行 24 簇导出表。Meta 组织的两个交换文件的 SHA-256 分别为
`a827f7c35ad500adce98399e2b00b7a370a709ad13ad6a1fa2548c612c31f395`（下行）和
`82a9de8aa48a36b4101198973bf9db46674f0805d9fa1f87e643299e620994b3`（上行）。
在两个方向各 38 个源时刻，以 −5、−2.5、0、2.5、5 MHz 频偏运行命令行。
独立复核程序从 CSV 的损耗、相位、时延直接计算，并用单独的时间表核对每帧身份。
380 个复数值通过；最大绝对数值差下行为 1.72×10⁻²⁰，上行为 1.54×10⁻²⁰ 以内。
两方向末帧均为 37，时刻为 36.17493937675094 秒，载频处下行复数为
(1.6951162149620494e-7, −6.097723221664496e-7)，上行为
(1.7287273864062641e-7, −6.314623721627944e-7)。这些误差只描述两种软件计算的一致性。

复现时先用前述命令生成单方向输出，再运行下例，将文件名替换为该方向实际文件。

```bash
python scripts/verify_sampled_export.py output.json R0B0S0D_cluster_cdl.csv independent-carrier-check.csv --direction downlink --offsets-hz -5000000 -2500000 0 2500000 5000000
```

导出缺少任务编号或请求预算时，来源 taskId 和 requestedClusterCount 分别显式填 null。
mpdbPath、每帧链路身份和实际簇数仍必填。未知预算仅按本版上限 24、输入路径数及已知有效路径数校验；
调用者应从原始输入保留预算及其来源，不能从实际簇数反推请求参数。
