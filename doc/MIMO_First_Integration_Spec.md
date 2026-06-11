# MIMO-First ↔ Channel Engine 集成契约 (v2.0)

**Status**: ACTIVE (生效 2026-05-19, 更新 2026-06-11 audit batch 2 后 — 升级注意事项见 §0)
**Supersedes**: v1.0 (2026-04, 见 §6 历史)
**Library version**: Phase 8 / P2-7 + audit batch 2 (PR #16–#31)

---

## 0. ⚠️ Audit batch 2 升级注意事项 (2026-06-11, PR #16–#31)

> ChannelEngine 完成了一轮 16-PR 的 TR 38.901 合规修复 (silent-ignored
> config / spec 简化全面清理, 实现状态总表见
> [`docs/spec/tr38901-section75-traceability.md`](../docs/spec/tr38901-section75-traceability.md))。
> **本节是 MIMO-First/MIMO-3D 调用方升级前必读。**

### 0.1 接口层面: 零破坏

§2 推荐路径的全部 public API 签名不变 (`CustomCDLProfile.from_dict/from_file`、
`TargetChannelConfig`/`ChamberConfig`/`AntennaArrayConfig`、
`MIMO_OTA_Simulator().run()`、`PropsimASCIIExporter.export()`); schema 无新增
必填字段。被删除的 `mimo_ota_simulator/parsers.py`(import 即 ImportError)、
`channel_model_38901` 的 `ChannelProfile` 类、两份备用参数 JSON 均为
零引用 dead code, 外部不可能在用。

### 0.2 新增 fail-fast(唯一可能让调用方报错的两处)

| 配置 | 旧行为 | 新行为 (PR #20) |
|---|---|---|
| 单极化 chamber (`dual_polarized=False`) + `tx_antenna.polarization` ≠ `'V'` | 静默按 V 处理 | `ValueError`(与 rx 侧既有校验对齐) |
| `cluster_model_name` 不在白名单(如小写 typo `'cdl-a'`) | 静默落入随机簇路径 | `ValueError` |

这两类配置此前拿到的本来就是错误信道; 升级后请全量跑一遍调用路径确认无触发。

### 0.3 数值输出变化(无错无警, 但结果不同 — 均为修复)

**custom CDL 路径**(MIMO-First 主路径):
- 探头权重改 (az, el) 各向异性高斯 (PR #23): profile 中 `zoa_deg≠90` 或
  custom 立体 probe 布局时 A 矩阵变化; **ring 布局 + 水平簇 (zoa=90°)
  与旧版 bit-identical**。
- `probe_phase_jitter=True` 时导出 .asc 现在真含 jitter (PR #22) — 此前
  导出与 `effective_H_matrix` 静默不一致。
- SoS/FPGA 导出 (PR #27): ray 随机配对 (§7.5 step 8) + **`initial_phases_rad[0]`
  真进 per-ray 相位** — 此前调用方传的初相在 SoS 导出里被静默丢弃。
- 探头 `pattern='TR38.901'` 现在真衰减 B 矩阵 (PR #19, 峰值 0 dB 形状因子,
  此前等同 Isotropic)。

**standard 3GPP 路径**(`model_name='UMa'` 等): 统计特性全面变化 — LSP
互相关激活 (PR #16)、zenith 约定统一 (PR #17)、角度围绕几何 LOS 方向 +
C_φ/C_θ 表 (PR #25)、per-cluster shadowing ζ 按 Table 7.5-6 (PR #26)、
-25 dB 弱簇删除 (PR #26)、簇内 ray 扇出缩小约一个量级 (PR #24)。
**同 seed 输出与旧版必然不同; 任何固定 seed 的 golden 基线必须重生成。**

### 0.4 无害新增

- 新告警: `num_rays≠20`、单极化 + `xpr_db`/`initial_phases_rad`、参数文件
  缺新字段 (回退旧行为 + 升级提示)。
- 结果 dict 新键: `h_subcarriers`(WiFi+OFDM, per-antenna `(n_fft, N_rx, N_tx)`,
  PR #28)、`probe_phase_jitter_vector_ul`(PR #22)。
- `PropsimASCIIExporter.last_mode_a_info`: Mode A 链路预算
  (`required_attenuation_db` 含 `target_output_power_dbm`, PR #21);
  Mode A 的 stdout 打印格式有变, 勿解析 stdout, 改读该属性。

### 0.5 调用方升级清单

1. 确认走的是真服务还是 mock(§5 adapter 已于 2026-05-18 修复并于 2026-06-11
   对 batch 2 实证, 但 `MOCK_ASC_MODE=1` 的部署仍出占位 .asc — 查响应
   `mock_mode` 字段);**库更新后必须重启 channel-engine-service**(sys.path
   惰性 import + 模块缓存, 长跑进程可能持旧库);
2. 升级后重跑端到端回归; 固定 seed 的 golden 基线重生成;
3. 检查配置是否触发 §0.2 的两个 ValueError;
4. 若消费 SoS 导出: 初相现在生效、射线配对随机化 — 行为修复而非回归。

---

## 1. 架构定位与边界

| Component | Role | Scope |
|---|---|---|
| **MIMO-First** (Client / Orchestrator) | 数字大脑：管硬件、跑 ray-tracing、组装 HTTP payload | MIMO-First repo |
| **channel-engine-service** | HTTP wrapper：接 POST、转 dict、调本 repo library、打包返回 | MIMO-First repo `channel-engine-service/` 微服务 |
| **Channel Engine library** (本 repo) | 数学编译器：CDL → OTA 权重 → ASC 文件 | 本 repo (ChannelEgine) |

**重要边界**（per strict-pfs-rollout Decision D11 + Phase 5 Q1.A）:
- **本 repo 不实现** HTTP service / FastAPI endpoint / Zip 打包
- **本 repo 提供** library-level Python API (`CustomCDLProfile` Pydantic schema)
- MIMO-First 微服务负责所有 HTTP 适配 + 序列化 + 编排

---

## 2. Library-level contract (本 repo authoritative)

### 入口 1: `CustomCDLProfile.from_dict(payload_dict)`

```python
from mimo_ota_simulator.cdl_schema import CustomCDLProfile

profile = CustomCDLProfile.from_dict({
    'center_frequency_hz': 2.6e9,
    'pathloss_db': 88.5,
    'ue_velocity_mps': [4.17, 0.0, 0.0],   # m/s, 3-vector (NOT kph)
    'is_los': True,
    'k_factor_db': 10.0,                    # Optional. Applied to LOS first cluster.
    'clusters': [
        {
            'delay_s': 0.0,
            'power_linear': 1.0,
            'aoa_deg': 0.0, 'aod_deg': 0.0,
            'zoa_deg': 90.0, 'zod_deg': 90.0,
            'as_aoa_deg': 1.0, 'as_aod_deg': 0.5,
            'xpr_db': 12.0,                                # Optional. Phase 5: plumb only; Phase 6: dual-pol.
            'initial_phases_rad': [0.785, 0.0, 0.0, 0.0],  # Optional. [Θθθ, Θθφ, Θφθ, Θφφ] in radians.
        },
        # ... NLOS clusters
    ],
})
```

### 入口 2: `CustomCDLProfile.from_file(json_path)`

```python
profile = CustomCDLProfile.from_file('/tmp/cdl.json')
```

JSON 文件内容 = `from_dict` 的 dict 一对一。

### 喂给 Simulator

```python
from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, TargetChannelConfig, PropsimExportConfig,
)
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.exporters import PropsimASCIIExporter

config = TargetChannelConfig(
    input_mode='custom',
    custom_profile=profile,                 # 关键：Pydantic 对象直接传，不需要写临时文件
    tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
    rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
)
chamber = ChamberConfig(
    num_probes=16, distribution='ring', radius_m=1.5,
    dual_polarized=True,
    probe_antenna={'gain_dBi': 8.5, 'pattern': 'Isotropic'},
    # P2-7 non-ring layout (sparse / sector / dual-ring): set
    # distribution='custom' and pass probe_positions=[ProbePosition(az,el), ...]
    # See §3.2 below.
)
sim = MIMO_OTA_Simulator()
result = sim.run(chamber, config, synthesis_method='strict_pfs')

# Export DL ASC (for Keysight F64 / Propsim hardware)
cfg = PropsimExportConfig(filename='out.asc', mode='B',
                          duration_s=0.1, sample_rate_hz=1000)
PropsimASCIIExporter(cfg).export(result, direction='dl')
```

---

## 3. Schema 完整字段表

### Top-level (`CustomCDLProfile`)

| Field | Type | Required | Notes |
|---|---|---|---|
| `center_frequency_hz` | float (>0) | ✅ | Hz |
| `pathloss_db` | float | ✅ | 自由空间 + 阴影 |
| `ue_velocity_mps` | Tuple[float, float, float] | ✅ | m/s 3-vector，v1.0 是 kph 已废弃 |
| `is_los` | bool | ✅ | |
| `k_factor_db` | Optional[float] | ❌ | LOS scenario 下 redistribute 第一 cluster 功率 (TR 38.901 §7.5.6) |
| `clusters` | List[CDLClusterSpec] (min 1) | ✅ | |

### Per-cluster (`CDLClusterSpec`)

| Field | Type | Required | Phase 5 status |
|---|---|---|---|
| `delay_s` | float (≥0) | ✅ | Consumed |
| `power_linear` | float (>0) | ✅ | Consumed (normalized to sum 1) |
| `aoa_deg` | float | ✅ | Consumed |
| `aod_deg` | float | ✅ | Consumed |
| `zoa_deg` | float (default 90) | ❌ | Consumed —— Tx/Rx steering、Doppler、TR38.901 element gain、**probe 权重映射 (az,el) 各向异性高斯**(2026-06-10 起;此前权重 azimuth-only) |
| `zod_deg` | float (default 90) | ❌ | Consumed —— Tx steering / element gain(probe 映射只看到达侧 `zoa`) |
| `as_aoa_deg`/`as_aod_deg`/`as_zoa_deg`/`as_zod_deg` | float (≥0, default 0) | ❌ | Consumed —— ray 角度扇出;`as_aoa`/`as_zoa` 另作 probe 权重高斯的 az/el 宽度 |
| `xpr_db` | Optional[float] | ❌ | **Phase 6: 真用** —— 进 dual-pol pol_matrix off-diag sqrt(1/κ) (TR 38.901 §7.3.2). 仅 chamber.dual_polarized=True 时生效；single-pol 路径忽略此字段。Caller 没设值时默认 7 dB + warning |
| `initial_phases_rad` | Optional[List[float, len=4]] | ❌ | **Phase 6: 全 4 个都真用**: 索引 0 (Θθθ) per-cluster base phase 进 strict_pfs，索引 1-3 (Θθφ, Θφθ, Θφφ) 进 dual-pol pol_matrix |
| `num_rays` | int (>0, default 20) | ❌ | 当前 strict_pfs/ray 实现忽略此字段固定 20；Phase 7+ 可配 |

`extra='forbid'`: 多余字段（拼错的、v1.0 残留字段）会被 Pydantic ValidationError 拒绝。常见坑：`pwr_linear` (typo)、`global_params` (v1.0 嵌套)、`ray_tracing_channel` (v1.0 嵌套)。

---

## 3.2 Non-Ring 暗室支持 (Phase 8 / P2-7, 2026-05-19)

`ChamberConfig.distribution`: 现在支持 `'ring'` (默认) / `'multi-ring'` / `'custom'`。
非 ring 配置（sparse / sector / dual-ring）必须显式传 `probe_positions`：

```python
from mimo_ota_simulator.data_models import ChamberConfig, ProbePosition

chamber = ChamberConfig(
    num_probes=4, radius_m=1.5, distribution='custom',  # 'multi-ring' 同
    probe_positions=[
        ProbePosition(azimuth_deg=10.0),                       # 全水平面
        ProbePosition(azimuth_deg=95.0),
        ProbePosition(azimuth_deg=180.0, elevation_deg=15.0),  # 仰角 +15°
        ProbePosition(azimuth_deg=270.0, elevation_deg=-15.0), # 俯角 -15°
    ],
    dual_polarized=False,
    probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
)
```

**MIMO-First 微服务侧 schema 扩展建议**:
```jsonc
"chamber_config": {
  "num_probes": 8,
  "radius_m": 2.0,
  "dual_polarized": true,
  "distribution": "ring",           // OR "custom" / "multi-ring"
  "probe_positions": [               // P2-7: 仅 distribution != "ring" 时需要
    { "azimuth_deg": 0.0,   "elevation_deg": 0.0 },
    { "azimuth_deg": 90.0,  "elevation_deg": 0.0 },
    // ... 每 probe 一个
  ]
}
```

**MIMO-First DB 已有的 `Probe.position: {azimuth, elevation}` 字段** 直接 map 到这。
其他规则:
- `distribution='ring'` (默认): `probe_positions` 可省略，走 `(p × 360°/N)` 等距公式
- `distribution='custom'` / `'multi-ring'`: `probe_positions` 必须，且 `len() == num_probes`
- `radius_m` 仍 chamber-wide uniform (MIMO-First DB 也没存 per-probe radius)

**触发场景** (per MIMO-First architecture note):
- PWS 工程 (sector geometry)
- 非标暗室 (sparse layout 省 probe / dual-ring 增强 elevation)
- 当前 lab ring 8-probe 不触发此 path; ring 默认 backward compat 不变

详见 [`docs/api/mimo_ota_simulator.md`](../docs/api/mimo_ota_simulator.md) §9。

---

## 3.1 Dual-Polarization 支持 (Phase 6+)

Phase 6 实现了 dual-polarization synthesizer，consume XPR + cross-pol initial phases。

**关键 schema 改动**:
- `AntennaArrayConfig.polarization: Literal['V', 'H'] = 'V'` 新增字段。默认 'V' 保 backward compat。
- Slant ±45° / 任意角 polarization 是 Phase 7 (deferred)，Phase 6 拒绝这些值。

**MIMO-First micro-service caller 适配建议**:
- 单极化 DUT (绝大多数 LTE/5G FR1)：caller 不动，schema 默认即可
- 双极化 DUT：caller 设 `chamber.dual_polarized=True` + 每 cluster 提供 `xpr_db` + `initial_phases_rad[4]`
- DUT 是 H-pol：caller 设 `rx_antenna.polarization='H'`（同时必须 `chamber.dual_polarized=True`）

**详细 behavior matrix**: 见 `docs/api/mimo_ota_simulator.md` §8。

---

## 4. Synthesis method 选择

| `synthesis_method` | 推荐场景 | strict PFS contract |
|---|---|---|
| `'strict_pfs'` | **推荐**：希望避免 OTA 输出端相位校准；first-call / low-Doppler 精度敏感 | ✅ per-realization E[A_i·A_j*]=0 |
| `'ray'` | 大平均统计验证（与 `probe_phase_jitter=True` 合用） | ⚠️ ensemble 极限 |
| `'cluster'` | **DEPRECATED** | rank-1 outer product，不能 strict PFS |
| `'pinv'` | **DEPRECATED, F64 broken** | 不产 wideband impulse response |

MIMO-First 默认应用 `'strict_pfs'`。

---

## 5. MIMO-First 微服务侧 wrapper 责任（reference, 不是本 repo 范围）

为完整画面，MIMO-First 在 `channel-engine-service` 那侧需要：

1. **HTTP endpoint**: 比如 `POST /api/v1/synthesize_hardware_pipeline`
2. **Payload 解析**: 把 HTTP JSON → 调 `CustomCDLProfile.from_dict(payload['cdl'])`
3. **Chamber + simulation_rules 解析**: 构造 `ChamberConfig`、`TargetChannelConfig`、calibration
4. **调用 library**: `MIMO_OTA_Simulator().run(...)` + `PropsimASCIIExporter().export(...)`
5. **打包**: 把生成的 `.asc` 文件 zip + Base64 + 组 response JSON

**~~已知 gap~~ → 已修复 (2026-05-18 MIMO-First P0-7)**: 旧版 `hardware_pipeline.py`
调本 repo 接口三层全错（类名 `OTASimulator`、构造签名、`run_with_external_clusters`
不存在）→ 已按本文档 §2 重写：`CustomCDLProfile.from_dict` + `ChamberConfig`
(probe_positions 透传) + `TargetChannelConfig` (custom/standard 分路) +
`MIMO_OTA_Simulator().run(synthesis_method='strict_pfs')` +
`PropsimASCIIExporter.export_to_zip_memory` base64 直出；mock 仅显式
`MOCK_ASC_MODE=1`，不再静默 fallback。**2026-06-11 对 audit batch 2 (b1c2eca)
实证**: MIMO-First `test_channel_engine_real_path.py` 13/13 过 + CAICT-FS
multi-ring 31 位置（含仰角簇、全四 `as_*` 字段）经适配层真合成 124 .asc OK；
`as_aod/as_zoa/as_zod` 端到端透传同日补齐 (MIMO-First #164)。

---

## 6. v1.0 历史 + 主要变化

v1.0 (2026-04) 的设计有几个问题：
- 顶层结构嵌套（`ray_tracing_channel.clusters`）跟实际 JSON 实现不一致
- `ue_velocity_kph`（scalar 或 vector，混乱）vs v2.0 `ue_velocity_mps`（强类型 3-vector）
- 缺 XPR / K-factor / per-ray init phases 字段
- 设计上把 `chamber_config / calibration_data` 跟 CDL profile 混在一个 payload，难以 reuse；v2.0 让 caller 自由组合

v1.0 schema **library 层仍可读**（通过 `CustomJSONBuilder` 兼容 shim，emit DeprecationWarning），但**新代码请用 v2.0**。

| v1.0 字段 | v2.0 对应 | 备注 |
|---|---|---|
| `global_params.center_frequency_hz` | `center_frequency_hz` | 顶层化 |
| `global_params.pathloss_db` | `pathloss_db` | 顶层化 |
| `global_params.ue_velocity_kph` | `ue_velocity_mps` | 单位 kph → m/s，强类型 3-vector |
| `clusters[].power_relative_linear` | `clusters[].power_linear` | 重命名 |
| `is_los` | `is_los` | 一致 |
| (无) | `k_factor_db` | 新 |
| (无) | `clusters[].xpr_db` | 新 |
| (无) | `clusters[].initial_phases_rad` | 新 |

---

## 7. 测试 / 引用

- 库 e2e: `tests/test_custom_cdl_e2e.py` (Phase 5 T5.8)
- Schema 单元: `tests/test_cdl_schema.py` (T5.1, T5.4, T5.5)
- 库 API 文档: `docs/api/mimo_ota_simulator.md` §6 CustomCDLProfile
- 设计原始: `docs/plans/external-cdl-input-design.md`

---

> Library version 推进时本文档同步更新。MIMO-First 微服务侧适配请反引用此版本号。
