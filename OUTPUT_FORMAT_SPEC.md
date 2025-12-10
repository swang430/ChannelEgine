# 仿真结果导出文件格式说明

本文档详细描述了 `ChannelModel38901` GUI 导出的 JSON 结果文件的结构和字段含义。

## 文件结构概览

导出文件是一个标准的 JSON 对象，包含以下顶层字段：

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `timestamp` | string | 仿真完成的时间戳 (ISO 8601 格式) |
| `pathloss_db` | float | 计算得出的大尺度路径损耗 (dB) |
| `is_los` | boolean | 信道条件是否为视距 (True: LOS, False: NLOS) |
| `lsps` | object | 本次仿真生成的大尺度参数 (LSP) 集合 |
| `clusters` | array | 生成的多径簇列表，每个元素代表一条传播路径 |

---

## 详细字段定义

### 1. `lsps` (大尺度参数)

此对象包含所有随机生成或确定性加载的大尺度信道参数。

| 字段名 | 类型 | 单位 | 描述 |
| :--- | :--- | :--- | :--- |
| `delay_spread_s` | float | 秒 (s) | 均方根时延扩展 (DS) |
| `departure_azimuth_spread_deg` | float | 度 (°) | 离开方位角扩展 (ASD) |
| `arrival_azimuth_spread_deg` | float | 度 (°) | 到达方位角扩展 (ASA) |
| `departure_zenith_spread_deg` | float | 度 (°) | 离开天顶角扩展 (ZSD) |
| `arrival_zenith_spread_deg` | float | 度 (°) | 到达天顶角扩展 (ZSA) |
| `shadow_fade_db` | float | dB | 阴影衰落值 |
| `k_factor_linear` | float | - | 莱斯因子 (线性值)，仅在 LOS 场景下有效 |

### 2. `clusters` (多径簇列表)

这是一个对象数组，每个对象描述了一个多径簇的物理属性。

| 字段名 | 类型 | 单位 | 描述 |
| :--- | :--- | :--- | :--- |
| `delay` | float | 秒 (s) | 该簇的到达时延 |
| `power` | float | 线性值 | 该簇的归一化功率 |
| `aod_phi` | float | 度 (°) | 离开方位角 (Azimuth of Departure) |
| `zod_theta` | float | 度 (°) | 离开天顶角 (Zenith of Departure) |
| `aoa_phi` | float | 度 (°) | 到达方位角 (Azimuth of Arrival) |
| `zoa_theta` | float | 度 (°) | 到达天顶角 (Zenith of Arrival) |
| `k_n` | float | 线性值 | 交叉极化功率比 (XPR) 的线性值 |
| `xpr_db` | float | dB | 交叉极化功率比 (XPR) 的对数值 |

---

## 示例数据

```json
{
    "timestamp": "2023-10-27T10:30:00.123456",
    "lsps": {
        "delay_spread_s": 3.24e-07,
        "departure_azimuth_spread_deg": 15.2,
        "arrival_azimuth_spread_deg": 22.5,
        "departure_zenith_spread_deg": 3.1,
        "arrival_zenith_spread_deg": 5.4,
        "k_factor_linear": 0.0,
        "shadow_fade_db": -2.5
    },
    "pathloss_db": 105.4,
    "is_los": false,
    "clusters": [
        {
            "delay": 1.2e-07,
            "power": 0.45,
            "aoa_phi": 45.0,
            "zoa_theta": 92.0,
            "aod_phi": -30.0,
            "zod_theta": 88.0,
            "xpr_db": 8.5,
            "k_n": 7.07
        },
        {
            "delay": 2.5e-07,
            "power": 0.22,
            "aoa_phi": -10.0,
            "zoa_theta": 85.0,
            "aod_phi": 15.0,
            "zod_theta": 95.0,
            "xpr_db": 7.0,
            "k_n": 5.01
        }
    ]
}
```
