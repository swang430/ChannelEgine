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
