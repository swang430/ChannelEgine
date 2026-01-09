# WiFi 802.11ax 互易信道 MIMO OTA 设计方案

## 1. 目标与范围
本方案面向在现有信道仿真与 OTA 框架中新增 WiFi（以 802.11ax/TGax 为主）互易信道 MIMO OTA 能力，重点覆盖：
- 室内 TGax Model A–F 的 Tap-Cluster 信道建模
- UL/DL 同时导出与互易性建模
- 可选 OFDM 子载波级别 H_subcarriers 计算
- 保持与现有 OTA 映射、导出与 UI 体系兼容

非目标（可延后）：
- 11ac/11n 全套支持
- 11be/60GHz 复杂场景
- 天线方向图与极化的完整 3D 建模

## 2. 术语与分层对齐
### 2.1 38.901 vs WiFi 的分层差异
- 38.901 CDL：Cluster = delay + power + mean angles + spreads；Ray = 角度偏移
- WiFi TGn/TGax：Tap = delay + power；Cluster = angles + spreads

### 2.2 统一合成单元（TapComponent）
为复用现有 OTA 合成逻辑，引入“TapComponent”概念：
- TapComponent = Tap（delay, power） + Cluster（AoA/AoD/AS）
- TapComponent 是“带时延的空间分量”，可进行 ray 展开

结论：WiFi 模型在合成阶段表现为“Ray-Tap”组合，取代 38.901 的“Ray-Cluster”组合，但 Tap 仍保留独立时延结构。

## 3. 架构设计
### 3.1 新增模块
建议新增独立包以避免与 38.901 混杂：
- `wifi_channel_model/`
  - `simulator.py`（WiFi 顶层 API）
  - `models/`（TGax Model A–F）
  - `parameters/`（JSON 参数表）
  - `common/`（Tap/Cluster 数据结构）

### 3.2 与 OTA 集成
`mimo_ota_simulator` 增加标准选择：
- `standard: "3gpp" | "wifi"`
- 当 standard=wifi 时，从 `wifi_channel_model` 获取仿真结果

保持 OTA 输出结构一致（至少包含）：
- `channel_matrix`（中心频点 H）
- `channel_impulse_response`（tap 列表）
- `clusters` 或 `tap_components`（用于可视化/调试）

## 4. 参数与数据结构
### 4.1 WiFi 模型配置
新增配置字段（示例）：
```
WifiChannelConfig:
  model_name: "TGax-A" | "TGax-B" | ...
  center_frequency_hz: float
  bandwidth_hz: float
  enable_ofdm: bool
  n_fft: int
  sample_rate_hz: float
  reciprocity_mode: "ideal" | "calibrated" | "uncalibrated"
  reciprocity_transform: "transpose" | "conjugate_transpose"
```

### 4.2 参数表格式（建议 JSON）
关键点：避免“延迟列 + power 序列”信息丢失，采用显式 (delay, power) 对。
```
{
  "model": "TGax-D",
  "delay_columns_ns": [0, 10, 20, ...],
  "clusters": [
    {
      "cluster_id": 0,
      "aoa_deg": 158.9,
      "aod_deg": 332.1,
      "as_rx_deg": 27.7,
      "as_tx_deg": 27.4,
      "taps": [
        {"delay_ns": 0, "power_db": 0.0},
        {"delay_ns": 10, "power_db": -0.9},
        ...
      ]
    }
  ]
}
```

### 4.3 带宽扩展（TGac/TGax 规则）
当带宽超过 40 MHz，tap 间隔按扩展因子缩放：
- 40–80 MHz：间隔 5 ns
- 80–160 MHz：间隔 2.5 ns
- 160–320 MHz：间隔 1.25 ns
（可由 `bandwidth_hz` 推导）

## 5. 信道生成流程（WiFi）
### 5.1 TapComponent 构建
对每个 cluster 的每条 tap 生成 TapComponent：
- `delay = tap.delay`
- `power = tap.power`
- `aoa/aod/as` 继承 cluster 参数

### 5.2 合成模式
#### 5.2.1 Cluster 模式（轻量）
每个 TapComponent 视作平面波或相关矩阵贡献，按 tap 延迟叠加。
适合快速快照。

#### 5.2.2 Ray 模式（推荐）
每个 TapComponent 展开 N 条射线：
- Ray 角度从 `AS` 采样（高斯或固定 offset 表）
- 每条 Ray 共享 tap 延迟
- 叠加得到 tap 级系数
适合空间相关性验证与多普勒仿真。

### 5.3 OFDM 子载波计算（可选）
当 `enable_ofdm=True`：
```
H(f_k) = Σ_tap h_tap * exp(-j 2π f_k τ_tap)
```
输出：
- `H_subcarriers` 维度 `(N_subcarriers, N_rx, N_tx)`

## 6. 互易性与 UL/DL
### 6.1 互易变换
两种常用定义：
- `transpose`: `H_ul = H_dl^T`
- `conjugate_transpose`: `H_ul = H_dl^H`

### 6.2 硬件响应引入
物理信道 `H_p`：
- `H_dl = G_rx_dut * H_p * G_tx_probe`
- `H_ul = G_rx_probe * H_p^T * G_tx_dut`

模式说明：
- `ideal`: 忽略硬件响应
- `calibrated`: 先引入响应，再预失真补偿
- `uncalibrated`: 保留硬件误差影响

## 6.3 WiFi 场景与路径损耗现状说明
- TGn/TGax 的 Model A–F 主要描述小尺度多径（PDP/角度/簇结构），最大时延随 A→F 增大，可用于“环境尺度/复杂度”的定性选择。
- 目前 WiFi 路径损耗**仅采用自由空间损耗（FSPL）占位**，不绑定 A–F 场景，也未引入 TGax 建议的断点距离/分段指数/阴影衰落模型。
- 现状保持：`PL = 20*log10(d_m) + 20*log10(fc_hz) - 147.55`。

后续优化方向（待定）：
- 引入 TGax 建议的“断点距离 + 分段路径损耗指数 + 阴影衰落标准差”模型，并与 A–F 场景做可选绑定或独立选择。
- 若做系统级 WiFi 场景，可选复用 3GPP TR 38.901 的路径损耗/LOS 概率模型（当前仅用于 3GPP 标准分支）。

## 7. OTA 映射与导出
### 7.1 OTA 映射
- `pinv`: `A = pinv(B) @ H_target`（数值逼近）
- `cluster/ray`: 物理驱动合成（推荐）

WiFi 默认建议：`ray`，`pinv` 作为对照，`cluster` 作为轻量模式。

### 7.2 UL/DL 同时导出
导出双向文件：
- `propsim_dl_*.asc`
- `propsim_ul_*.asc`

文件头部需包含真实 `Carrier_Frequency` 与 `CIR_Update_Rate`。
默认采用保守策略：UL/DL 独立文件，设备若支持双向复用可只加载 DL 文件。

## 8. UI 与配置
在 `app.py` / `gui.py` 增加：
- 标准选择：3GPP / WiFi
- WiFi 模型选择：TGax A–F
- 带宽与 OFDM 开关
- 互易模式与变换方式
- UL/DL 同时导出开关（默认开启）

## 9. 测试与验证
新增测试建议：
- Tap-Cluster 对齐测试（显式 tap 列表）
- Ray-Tap 合成一致性测试
- UL/DL 互易性误差测试
- OFDM 子载波一致性测试（tap -> H_subcarriers）
- OTA 映射回归金样（UL/DL 各一套）

## 10. 风险与评估点
- Tap/Cluster 稀疏表对齐错误会导致 PDP 偏差
- 互易变换（T vs H）需与硬件基带约定一致
- OFDM 子载波计算开销高，需按开关控制

## 11. 文档与参考
- `doc/wifi/wifi信道模型梳理.docx`
- `doc/wifi/wifi信道模型跟踪.docx`
