# ChannelEngine - 3GPP TR 38.901 信道模型与MIMO OTA仿真器

## ⭐ Cross-project context (last updated 2026-05-19, post-Phase-8 / P2-7)

> 本节是 cross-repo coordination notes，**任何 AI session 或新加入的开发者进入本 repo 时应先读完这里**。
> 详细 strict-PFS rollout 完整 plan 见 [`docs/plans/strict-pfs-rollout.md`](docs/plans/strict-pfs-rollout.md)。
> 历史背景 + spec 引用见姊妹 repo [`MIMO-First`](https://github.com/swang430/Meta-3D) 下的
> [`docs/features/calibration/pfs-phase-immunity.md`](../MIMO-First/docs/features/calibration/pfs-phase-immunity.md)。

### 当前状态（2026-05-18, Phase 0-6 全部 merged）

Strict-PFS rollout (PR #1, #2, #3, #4, #5, #6) 完整落地。下面是关键事实表（替换 2026-05-17 outdated 版本）：

| 维度 | 状态 | 详情 |
|---|---|---|
| **`synthesis_method='strict_pfs'`** | ✅ 生产可用 (Phase 1) | per-(probe, cluster) 独立 fading；per-realization `E[A_i·A_j*]=0`；推荐用于规避输出端相位校准 |
| `synthesis_method='ray'` | ✅ 默认 (D9 保守 opt-in 策略) | 大平均统计场景；strict_pfs 经 1-2 release 验证后可切默认 |
| `synthesis_method='cluster'` | ⚠️ DEPRECATED (Phase 3) | rank-1 outer product，结构性不能 strict PFS；DeprecationWarning |
| `synthesis_method='pinv'` | ⚠️ DEPRECATED + F64 broken | 不产 wideband impulse response |
| **`probe_phase_jitter` ±180°** | ✅ Phase 0 UI 修正 | 跟 phase cal 互斥时 runtime warning + UI st.warning；strict_pfs 下 UI 自动 disable |
| **External CDL 输入** (Phase 5) | ✅ `CustomCDLProfile` Pydantic schema | `from_file` / `from_dict` 双入口；含 XPR / K-factor / per-ray init phases 字段；SPEC v2.0 [`doc/MIMO_First_Integration_Spec.md`](doc/MIMO_First_Integration_Spec.md) |
| **Dual-Polarization synthesizer** (Phase 6) | ✅ XPR + cross-pol init phases 真用 | `AntennaArrayConfig.polarization: Literal['V', 'H']`；TR 38.901 §7.3.2 per-cluster 2×2 pol matrix；Tx 投影在 A 矩阵，Rx 投影在 B 矩阵 |
| **Non-Ring 暗室** (Phase 8 / P2-7, 2026-05-19) | ✅ `ChamberConfig.distribution = custom` + `probe_positions` | 修 MIMO-First architecture note flagged "ring-only silent constraint"；sphere→Cartesian 公式 `(r·cos(el)·cos(az), r·cos(el)·sin(az), r·sin(el))`；ring 默认 backward compat 不变 |

### MIMO-First 集成路径（推荐写法 post-Phase-6）

```python
from mimo_ota_simulator.cdl_schema import CustomCDLProfile
from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, TargetChannelConfig, PropsimExportConfig,
)
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.exporters import PropsimASCIIExporter

# MIMO-First 的 Ray-Tracing 输出 → CustomCDLProfile dict
profile = CustomCDLProfile.from_dict({
    'center_frequency_hz': 2.6e9, 'pathloss_db': 88.5,
    'ue_velocity_mps': [4.17, 0.0, 0.0],   # m/s, 3-vector (NOT kph)
    'is_los': True,
    'k_factor_db': 10.0,                    # LOS boost per TR 38.901 §7.5.6
    'clusters': [{
        'delay_s': 0.0, 'power_linear': 1.0,
        'aoa_deg': 0.0, 'aod_deg': 0.0,
        'as_aoa_deg': 1.0, 'as_aod_deg': 0.5,
        'xpr_db': 7.0,                       # Phase 6 dual-pol 真用
        'initial_phases_rad': [0, 0.785, 1.57, 3.14],  # Phase 6 4 个都用
    }],
})

# Library-level call (本 repo public API, 详见 docs/api/mimo_ota_simulator.md)
config = TargetChannelConfig(
    input_mode='custom', custom_profile=profile,
    tx_antenna=AntennaArrayConfig(num_cols=2, polarization='V'),
    rx_antenna=AntennaArrayConfig(num_cols=2, polarization='H'),  # H DUT
)
chamber = ChamberConfig(
    num_probes=16, distribution='ring', radius_m=1.5,
    dual_polarized=True,  # 必须 True 才能用 H/V/XPR
    probe_antenna={'gain_dBi': 8.5, 'pattern': 'Isotropic'},
)
result = MIMO_OTA_Simulator().run(chamber, config, synthesis_method='strict_pfs')

# F64 ASC export
PropsimASCIIExporter(PropsimExportConfig(
    filename='out.asc', mode='B', duration_s=0.1, sample_rate_hz=1000,
)).export(result, direction='dl')
```

### 已知 MIMO-First 侧 gap（**不在本 repo 范围**，需要 MIMO-First 那边修）

MIMO-First `channel-engine-service/app/api/endpoints/hardware_pipeline.py` 的接口跟本 repo 三层全错（Phase 3 调研 + Phase 5 §2 SPEC v2.0 详述）：
1. 类名错：`OTASimulator` (本 repo 实际叫 `MIMO_OTA_Simulator`) → ImportError
2. 构造签名错：本 repo `MIMO_OTA_Simulator()` 是空构造 → TypeError
3. 方法名错：本 repo 没有 `run_with_external_clusters` (D11 决定不实现) → AttributeError

实际行为：fallback 到 mock 模式 → MIMO-First 用户得到**假 .asc**。**修法**：MIMO-First 侧按上面"推荐写法"重写微服务 adapter。本 repo 决定 (D11) 不实现 `run_with_external_clusters` 等别名 —— `MIMO_OTA_Simulator.run()` 是 stable public API，service 侧适配责任在 MIMO-First。

### 未来 roadmap

| Phase | 状态 | 内容 |
|---|---|---|
| 0 / 1 / 2 / 3 / 5 / 6 | ✅ DONE | 见 `docs/plans/strict-pfs-rollout.md` |
| 4 | ✅ DONE | Cross-repo sync (本文档 + Meta-3D issue #55) |
| 7 | ⏸️ DEFERRED | Slant ±45° / 任意角 DUT polarization (memory: `project_phase7_slant_pol_roadmap.md`) |
| 8 (P2-7) | ✅ DONE (2026-05-19) | Non-ring chamber probe geometry; MIMO-First architecture note 推动 |

### Cross-project memory (machine-local)

本机 Claude session 在本 repo 工作时自动加载 [`~/.claude/projects/-Users-Simon-Tools-ChannelEgine/memory/MEMORY.md`](~/.claude/projects/-Users-Simon-Tools-ChannelEgine/memory/MEMORY.md)。关键 memory:
- `project_mimo_first_pfs_investigation.md` — 历史 PFS 调研 + 当前状态指针
- `project_strict_pfs_plan_locked.md` — 跟 swang430 对齐后锁定的 rollout 决策
- `project_phase7_slant_pol_roadmap.md` — Phase 7 slant pol deferred 触发条件

详细 spec 引用：TR 38.901 V19.0.0 §7.3.2 / §7.5.6 / §7.7-2；TR 37.977 V14.3.0 / V18.0.0 / V19.0.0 Annex F.2 + B.1；CTIA 01.40 V6.0.1 §2.2.2.3 / §4.2.5。

---

## 项目概述

**项目名称**：ChannelEngine
**版本**：v1.0 (Integration Branch)
**标准依据**：3GPP TR 38.901 V19.0.0
**主要功能**：5G/Beyond 5G信道建模 + MIMO OTA测试仿真

### 核心价值
本项目实现了基于3GPP TR 38.901标准的高精度信道模型生成引擎，并提供MIMO OTA（Over-The-Air）暗室测试仿真能力，支持从理论信道模型到物理测试环境的完整工作流。

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户接口层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   gui.py     │  │   app.py     │  │ CLI Scripts  │          │
│  │  (TKinter)   │  │ (Streamlit)  │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────┬──────────────┬──────────────────────────────────────┘
             │              │
             ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           mimo_ota_simulator/                            │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │   │
│  │  │ simulator  │  │data_models │  │  parsers   │         │   │
│  │  │   .py      │  │    .py     │  │    .py     │         │   │
│  │  └────────────┘  └────────────┘  └────────────┘         │   │
│  │       ↓ 依赖                                              │   │
│  └───────┼──────────────────────────────────────────────────┘   │
└──────────┼──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   核心引擎层 (Core Engine)                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           channel_model_38901/                           │   │
│  │  ┌────────────┐  ┌────────────────┐  ┌──────────────┐   │   │
│  │  │ simulator  │  │channel_generator│  │   models/    │   │
│  │  │   .py      │  │      .py        │  │  (7 scenarios)│   │
│  │  │ (Facade)   │  │                 │  │              │   │
│  │  └────────────┘  └────────────────┘  └──────────────┘   │   │
│  │  ┌────────────┐  ┌──────────────┐                       │   │
│  │  │  common/   │  │ parameters/  │                       │   │
│  │  │data_models │  │parameter_load│                       │   │
│  │  └────────────┘  └──────────────┘                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    测试验证层 (Testing Layer)                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           tests/                                         │   │
│  │  ┌─────────────────┐  ┌──────────────────┐              │   │
│  │  │test_lsp_statistics│ │statistical_utils │              │   │
│  │  │      .py        │  │      .py         │              │   │
│  │  │ (统计验证)      │  │  (工具函数)      │              │   │
│  │  └─────────────────┘  └──────────────────┘              │   │
│  │  ┌─────────────────┐  ┌──────────────────┐              │   │
│  │  │  test_*.py      │  │   cache/         │              │   │
│  │  │ (单元/集成测试)  │  │ (样本缓存)       │              │   │
│  │  └─────────────────┘  └──────────────────┘              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心模块设计

### 1. channel_model_38901 (核心信道引擎)

#### 1.1 职责
实现3GPP TR 38.901标准定义的随机信道模型，生成符合统计特性的MIMO信道矩阵。

#### 1.2 架构模式

**Facade模式 + Strategy模式**

```python
ChannelSimulator (Facade)
    ↓
ChannelGenerator (Core Engine)
    ↓
ABC_ChannelConditionModel ← UMa_ChannelCondition
ABC_PathlossModel         ← UMa_Pathloss
ABC_FadingModel           ← UMa_Fading
    ↓
ParameterLoader (Configuration)
```

#### 1.3 支持的场景

| 场景代码 | 全称 | 应用环境 |
|---------|------|---------|
| UMa | Urban Macro | 城市宏蜂窝 |
| UMi-StreetCanyon | Urban Micro (Street Canyon) | 城市微蜂窝（街道峡谷） |
| UMi-OpenArea | Urban Micro (Open Area) | 城市微蜂窝（开阔区域） |
| RMa | Rural Macro | 乡村宏蜂窝 |
| InH-Office | Indoor Hotspot (Office) | 室内热点（办公室） |
| SMa | Suburban Macro | 郊区宏蜂窝 |
| InF | Indoor Factory | 室内工厂 |

每个场景支持LOS和NLOS两种条件。

#### 1.4 关键数据结构

**LargeScaleParameters (LSP)**
```python
class LargeScaleParameters:
    delay_spread_s: float           # DS - 时延扩展
    departure_azimuth_spread_deg: float  # ASD - 出发方位角扩展
    arrival_azimuth_spread_deg: float    # ASA - 到达方位角扩展
    departure_zenith_spread_deg: float   # ZSD - 出发天顶角扩展
    arrival_zenith_spread_deg: float     # ZSA - 到达天顶角扩展
    k_factor_linear: float          # K - 莱斯因子
    shadow_fade_db: float           # SF - 阴影衰落
```

**EngineCluster (簇级参数)**
```python
class EngineCluster:
    delay: float        # 簇延迟
    power: float        # 簇功率
    aod_phi: float      # 出发方位角 (AoD)
    aoa_phi: float      # 到达方位角 (AoA)
    zod_theta: float    # 出发天顶角 (ZoD)
    zoa_theta: float    # 到达天顶角 (ZoA)
    k_n: float          # 交叉极化比 (XPR)
    h_matrix: ndarray   # 簇的H矩阵
```

#### 1.5 信道生成流程

```
1. 确定LOS/NLOS条件 (概率模型)
   ↓
2. 生成大尺度参数 (LSPs)
   - 从多元对数正态分布采样
   - 使用Cholesky分解实现相关性
   ↓
3. 生成簇级参数
   a. 簇延迟 (指数分布)
   b. 簇功率 (归一化)
   c. 簇角度 (AOA, AOD, ZOA, ZOD)
   d. 簇XPR
   ↓
4. 合成MIMO信道矩阵
   - 天线阵列响应
   - 多普勒频移
   - 路径损耗
   ↓
5. 返回 H_matrix, clusters, LSPs
```

#### 1.6 混合模型支持

**创新特性**：解耦大尺度环境与小尺度簇模型

```python
# 示例：UMa环境 + CDL-A簇模型
simulator = ChannelSimulator(
    scenario_name='UMa',           # 大尺度环境
    cluster_model_name='CDL-A',    # 小尺度簇模型
    center_frequency_hz=3.5e9
)
```

支持的簇模型：
- **Stochastic**: 随机生成（默认）
- **CDL-A/B/C/D/E**: 3GPP定义的确定性簇模型
- **SCME**: SCME标准簇模型

---

### 2. mimo_ota_simulator (MIMO OTA仿真器)

#### 2.1 职责
仿真MIMO OTA暗室环境，将理论信道模型映射到物理探头配置。

#### 2.2 核心概念

**OTA测试原理**：
```
目标信道 H_target (M_UE × M_BS)
    ↓
暗室合成: H_effective = B × A
    ↓
B矩阵: 探头到UE天线的响应 (M_UE × N_probe)
A矩阵: 基站到探头的激励权重 (N_probe × M_BS)
```

#### 2.3 两种合成方法

**1. Pseudoinverse方法 (pinv)**
```python
A = pinv(B) @ H_target
```
- 优点：精确逼近
- 缺点：可能产生非物理解

**2. Cluster方法 (cluster)**
```python
# 对每个簇单独处理
for cluster in clusters:
    w = gaussian_weight(probe_angle, cluster_angle, AS)
    A_cluster = w @ cluster.h_matrix
A_total = sum(A_cluster)
```
- 优点：物理可实现
- 缺点：近似误差

#### 2.4 物理参数模型

**探头环境**：
- 探头数量：1-64个
- 分布方式：环形/球形
- 极化：单极化/双极化
- 探头增益、校准向量

**暗室参数**：
- 探头环半径
- 静区半径
- 路径损耗模型
- 相位校准

#### 2.5 关键数据结构

```python
class ChamberConfig:
    num_probes: int
    radius_m: float
    dual_polarized: bool
    probe_antenna: ProbeAntennaConfig
    calibration_vector: Optional[ndarray]

class TargetChannelConfig:
    model_name: str              # UMa, UMi等
    cluster_model_name: str      # CDL-A等
    center_frequency_hz: float
    tx_antenna: AntennaArrayConfig
    rx_antenna: AntennaArrayConfig
```

---

### 3. 统计验证框架 (tests/)

#### 3.1 设计理念

**三层测试金字塔**：
```
      /\
     /  \  Comprehensive (1000样本, ~5min)
    /____\
   /      \
  / Standard\ (200样本, ~1min)
 /__________\
/            \
/    Quick    \ (50样本, ~10s)
/______________\
```

#### 3.2 验证内容

**LSP统计验证**：
- 均值检验 (t-test)
- 标准差检验 (Levene's test)
- 分布形状检验 (Kolmogorov-Smirnov test)
- 相关性矩阵验证

**Cluster级验证**（计划）：
- 延迟分布（指数分布）
- 功率分布
- 角度分布

#### 3.3 缓存机制

```python
# 样本缓存路径
tests/cache/{scenario}_{condition}_{n_samples}_fc{freq}_seed{seed}.pkl

# 示例
tests/cache/UMa_LOS_200_fc3.5_seed42.pkl
```

避免重复生成，加速测试。

#### 3.4 统计工具函数

**statistical_utils.py**：
- `calculate_lsp_statistics()` - 计算观测统计量
- `get_expected_lsp_params()` - 获取理论参数
- `test_mean_and_std()` - 均值/标准差检验
- `kolmogorov_smirnov_test()` - KS检验
- `calculate_correlation_matrix()` - 相关性计算

---

## 依赖关系

### 模块依赖图

```
mimo_ota_simulator
    ↓ (单向依赖)
channel_model_38901
    ↓
numpy, scipy, pydantic

tests
    ↓
channel_model_38901
mimo_ota_simulator (部分测试)
```

**无循环依赖** - 清晰的层次结构

### 外部依赖

```toml
[dependencies]
numpy = ">=1.20"
scipy = ">=1.7"
pydantic = ">=2.0"
matplotlib = ">=3.5"
pytest = ">=7.0"
tqdm = ">=4.60"
streamlit = ">=1.20"  # 仅app.py需要
```

---

## 数据流

### 端到端数据流示例

```
用户输入:
  scenario='UMa'
  fc=3.5GHz
  M_BS=4, M_UE=2
    ↓
ChannelSimulator.run()
    ↓
ChannelGenerator.generate_channel_realization()
    ↓
  1. 确定is_los=True (概率85%)
  2. 生成LSPs (DS=15ns, ASA=25°, K=8dB...)
  3. 生成12个cluster (delays, powers, angles)
  4. 合成H_matrix (2×4复矩阵)
    ↓
返回:
  H_matrix: (2, 4) complex128
  clusters: List[EngineCluster] (12个)
  lsps: LargeScaleParameters
  pathloss_db: -110.5
  is_los: True
    ↓
MIMO_OTA_Simulator.run()
    ↓
  1. 计算B矩阵 (2×32, 16探头双极化)
  2. cluster方法合成A矩阵 (32×4)
  3. 验证: H_effective ≈ H_target
    ↓
返回:
  A_matrix: (32, 4)
  B_matrix: (2, 32)
  validation_error: 0.15 dB
```

---

## 关键算法

### 1. 相关LSP采样 (Cholesky分解)

```python
# 相关矩阵C (6×6 for LOS)
C = [[1,    0.4,  0.8, ...],
     [0.4,  1,    0.5, ...],
     ...]

# Cholesky分解
L = cholesky(C)

# 采样独立高斯变量
z = randn(6)

# 生成相关变量
x = μ + σ ⊙ (L @ z)

# 转换回线性域
LSP = 10^x  (for DS, ASD, ASA, ZSA, K)
LSP = x     (for SF, already in dB)
```

### 2. 簇功率归一化(2026-06-11 audit batch 2 PR11 后的实际行为)

```python
# Step 1: 生成原始功率 (Z_n ~ N(0, ζ²), ζ 读 JSON per_cluster_shadowing_std_db)
P_n = exp(-τ_n × (r_τ - 1) / r_τ / DS) × 10^(-Z_n/10)

# Step 2: 归一化 (扩散功率, 不含 K)
P_n = P_n / sum(P_n)

# Step 3: -25 dB 弱簇删除, 剩余重新归一化 (§7.5 step 5 末)

# 注: LOS 的 K 因子不进簇功率本身 —— 引擎在 _synthesize_channel 做全局
# Rician 混合 sqrt(K/(K+1))·H_los + sqrt(1/(K+1))·H_nlos; K 调整功率
# (P_1 += K/(K+1)) 仅用于 step 6/7 的角度生成归一化 (eq 7.5-9/7.5-14)。
# LOS 的延迟另按 step 4 的 D(K) 多项式缩放 (功率生成用未缩放延迟)。
```

### 3. OTA cluster合成算法

```python
def calculate_weights_for_cluster(cluster, chamber):
    """为单个cluster计算探头权重"""
    weights = zeros(num_probes)

    for i, probe in enumerate(probes):
        # 高斯角度权重
        angle_diff = probe.angle - cluster.aoa
        weight = exp(-(angle_diff^2) / (2 * cluster.AS^2))
        weights[i] = weight

    # 归一化
    weights /= sum(weights)
    return weights

# 对所有cluster求和
A_total = sum(
    weights_i @ cluster_i.h_matrix
    for i, cluster_i in clusters
)
```

---

## 配置管理

### 参数文件结构

**parameters1013_tr38901_v19.0.0.json** (772行)

```json
{
  "standard_version": "19.0.0",
  "scenarios": {
    "UMa": {
      "lsp_parameters_los": {
        "mu_log10_DS": "-7.067 - 0.0794 * log10(fc_ghz)",
        "sigma_log10_DS": "0.57 + 0.026 * log10(fc_ghz)",
        "correlations": {
          "DS_vs_SF": -0.4,
          "ASA_vs_DS": 0.8,
          ...
        },
        "scaling_factors_aoa_aod": {...},
        "scaling_factors_zoa_zod": {...}
      },
      "pathloss_los": {...},
      "los_probability": {...}
    },
    ...
  },
  "cdl_models": {
    "CDL-A": [...],
    ...
  }
}
```

**关键改进**：
- 完整的scaling factors（v1024新增）
- 完整的ZSD参数（v1024新增）
- 符合TR 38.901 v19.0.0标准

---

## 测试策略

### 单元测试
```bash
pytest tests/test_uma_models.py
pytest tests/test_channel_generator.py
```

### 集成测试
```bash
pytest tests/test_mimo_ota_simulator.py
```

### 统计验证测试
```bash
# 快速冒烟测试
pytest tests/test_lsp_statistics.py -m quick

# 标准验证
pytest tests/test_lsp_statistics.py -m standard

# 全面验证
pytest tests/test_lsp_statistics.py -m comprehensive
```

### 回归测试
```bash
pytest tests/test_regression.py
```
使用golden文件确保数值稳定性。

---

## GUI应用

### gui.py (TKinter)
- **特点**：原生桌面应用
- **功能**：
  - 场景参数配置
  - 实时仿真运行
  - PDP/AoA/ZoA可视化
  - 结果导出

### app.py (Streamlit)
- **特点**：Web应用，易于部署
- **功能**：
  - MIMO OTA参数配置
  - 暗室设计验证
  - B矩阵/A矩阵计算
  - 实时结果展示

**运行**：
```bash
# TKinter
python gui.py

# Streamlit
streamlit run app.py
```

---

## 版本演进

### v1.0 - Integration Branch (当前)

**特性**：
- ✅ 7种场景支持 (UMa, UMi, RMa, InH, SMa, InF)
- ✅ 混合模型支持 (Scenario + CDL)
- ✅ MIMO OTA仿真器
- ✅ 统计验证框架
- ✅ 双GUI界面
- ✅ 完整参数文件 (v19.0.0)

**分支**：
- `main`: 稳定发布分支
- `integration-v1024`: 集成v1024增强的分支
- `feature/statistical-validation`: 统计验证测试分支

**提交历史**：
```
9990fdf - feat(tests): Add TR 38.901 statistical validation test suite
3d83617 - feat(integration): Merge non-breaking enhancements from v1024 branch
6f3a4d3 - feat: Add Chinese documentation and comments
b825464 - feat(ota_simulator): 实现基于物理的暗室和B矩阵模型
```

---

## 技术债务与未来工作

### 已知限制
1. **K-factor参数**：当前实现在log10域，已修正为dB/10转换
2. **SF快速测试失败**：50样本对σ=4.0分布样本量不足，属于预期行为
3. **v1024 API变更**：未合并的破坏性变更（H矩阵返回dict、射线级XPR）

### 计划增强
1. **统计验证扩展**：
   - 相关性矩阵验证
   - Cluster级统计测试
   - 空间相关性验证

2. **性能优化**：
   - 并行化信道生成
   - JIT编译 (Numba)
   - C++扩展

3. **功能扩展**：
   - 时变信道支持
   - 频率选择性信道
   - 多用户MIMO

4. **API演进**：
   - 全极化支持 (返回dict)
   - 射线级XPR (20值/簇)
   - 物理天线阵列响应

---

## 开发指南

### 环境设置
```bash
# 克隆仓库
git clone git@github.com:swang430/ChannelEgine.git
cd ChannelEgine

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 可编辑安装
pip install -e .
```

### 开发工作流
```bash
# 1. 创建特性分支
git checkout -b feature/my-feature

# 2. 开发并测试
pytest tests/

# 3. 提交代码
git add .
git commit -m "feat: Add my feature"

# 4. 推送并创建PR
git push origin feature/my-feature
```

### 代码规范
- **Python版本**：>=3.10
- **代码风格**：PEP 8
- **类型注解**：使用Pydantic for data models
- **文档字符串**：中文注释（面向中国团队）
- **测试覆盖率**：目标>80%

---

## 性能指标

### 信道生成性能
- **单次实现**：~10ms (UMa, 4×2 MIMO)
- **批量生成**：~50ms/100次 (使用缓存)
- **内存使用**：<50MB (单次)

### OTA仿真性能
- **B矩阵计算**：~5ms (16探头，2天线)
- **A矩阵合成**：~20ms (12簇，cluster方法)
- **总仿真时间**：~100ms (端到端)

### 测试性能
- **快速测试**：~10s (50样本)
- **标准测试**：~60s (200样本)
- **全面测试**：~300s (1000样本)

---

## 参考文献

1. **3GPP TR 38.901 V19.0.0**
   "Study on channel model for frequencies from 0.5 to 100 GHz"
   https://www.3gpp.org/specifications-technologies/releases/release-19

2. **CTIA OTA Test Plan**
   "Test Methodology for MIMO Multi-Probe Anechoic Chamber V7.0.0"

3. **相关学术论文**：
   - Spatial Correlation in MIMO Channels
   - OTA Testing Methodologies
   - Statistical Channel Models

---

## 许可证

**待定** - 请与项目维护者确认

---

## 贡献者

- **Simon Wang** (@swang430) - 项目维护者
- **Siyuan Yin** (@siyuanyin2002) - v1024分支贡献者
- **Claude Code** - AI辅助开发

---

## 联系方式

**GitHub**: https://github.com/swang430/ChannelEgine
**Issues**: https://github.com/swang430/ChannelEgine/issues

---

*最后更新：2025-11-12*
*版本：Integration v1.0*
