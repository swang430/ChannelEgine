
# 3GPP TR 38.901 Channel Model

This repository contains a Python-based implementation of the 3GPP TR 38.901 channel model for 5G and beyond.

## Features

- Implements the 3GPP TR 38.901 V19.0.0 standard.
- Supports major scenarios: UMa, UMi, RMa, InH, etc.
- **Hybrid Model**: Combine large-scale parameters (e.g., from UMa) with deterministic small-scale models (e.g., CDL-A) for physically meaningful simulations.
- **Deterministic LSPs**: Option to use median values for large-scale parameters for repeatable testing.
- Flexible and easy-to-use API.
- Included GUI for quick simulations and visualization.

## Installation

1. Clone the repository:
   ```
   git clone <repository_url>
   ```
2. Create a virtual environment:
   ```
   python3 -m venv .venv
   ```
3. Activate the virtual environment:
   ```
   source .venv/bin/activate
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Install the packages in editable mode:
   ```
   pip install -e ./channel_model_38901
   pip install -e ./mimo_ota_simulator
   ```

## Usage

### GUI

To run a simulation, you can use the provided GUI:

```
python gui.py
```

### Scripting

You can use the `ChannelSimulator` class in your own scripts. Here are a few examples:

**Basic Simulation**

```python
from channel_model_38901.simulator import ChannelSimulator

# Create a simulator instance for UMa scenario
simulator = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)

# Run the simulation
results = simulator.run()

# Print the results
print(f"Channel Matrix:\n{results['channel_matrix']}")
print(f"Pathloss: {results['pathloss_db']:.2f} dB")
```

**Hybrid Model Simulation**

Combine the UMa large-scale environment with a deterministic CDL-A cluster model.

```python
from channel_model_38901.simulator import ChannelSimulator

simulator = ChannelSimulator(
    scenario_name='UMa', 
    cluster_model_name='CDL-A',
    center_frequency_hz=3.5e9
)

results = simulator.run()

print(f"Channel Matrix (UMa + CDL-A):\n{results['channel_matrix']}")
```

**Simulation with Median LSPs**

Run a simulation with deterministic large-scale parameters.

```python
from channel_model_38901.simulator import ChannelSimulator

simulator = ChannelSimulator(
    scenario_name='RMa', 
    center_frequency_hz=0.7e9,
    use_median_lsps=True
)

results = simulator.run()

print(f"Channel Matrix (RMa with median LSPs):\n{results['channel_matrix']}")
```

---

## MIMO OTA Synthesis — Method Selection

`MIMO_OTA_Simulator.run(synthesis_method=...)` supports four methods. Pick using this decision tree:

```
Want strict per-realization PFS (E[A_i·A_j*]=0 without averaging)?
        │
        ├── YES → use 'strict_pfs'  (recommended; avoids output phase calibration)
        │
        └── NO  → mostly running statistical multi-drop validation?
                  │
                  ├── YES → use 'ray' (default)
                  │           ├── add probe_phase_jitter=True for ensemble decorrelation
                  │           └── jitter ↔ phase calibration are mutually exclusive
                  │                 (warning + UI guard in app.py)
                  │
                  └── NO  → consider whether single-snapshot deterministic suffices;
                            'cluster' / 'pinv' are deprecated, not recommended
```

| Method | Status | Use case |
|---|---|---|
| `'strict_pfs'` | ✅ Recommended (Phase 1) | Avoid output phase cal; first-call / low-Doppler precision; works with `phase calibration vector` (superposition, not mutex) |
| `'ray'` | ✅ Default | Statistical multi-drop validation; supports `probe_phase_jitter` for ensemble decorrelation |
| `'cluster'` | ⚠️ Deprecated | rank-1 outer product, structurally cannot represent strict PFS |
| `'pinv'` | ⚠️ Deprecated, F64 broken | Does not produce wideband impulse response |

See `docs/api/mimo_ota_simulator.md` for the full public API and field semantics.

### External CDL Input (Phase 5+)

For external callers (e.g., a ray-tracing pipeline producing custom CDL parameters), use the Pydantic-validated `CustomCDLProfile`:

```python
from mimo_ota_simulator.cdl_schema import CustomCDLProfile
from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
)
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator

profile = CustomCDLProfile.from_dict({
    'center_frequency_hz': 2.6e9, 'pathloss_db': 88.5,
    'ue_velocity_mps': [4.17, 0.0, 0.0],
    'is_los': True, 'k_factor_db': 10.0,
    'clusters': [
        {'delay_s': 0.0, 'power_linear': 1.0,
         'aoa_deg': 0.0, 'aod_deg': 0.0,
         'as_aoa_deg': 1.0, 'as_aod_deg': 0.5,
         'xpr_db': 7.0,
         'initial_phases_rad': [0, 0.785, 1.57, 3.14]},
    ],
})

config = TargetChannelConfig(input_mode='custom', custom_profile=profile, ...)
result = MIMO_OTA_Simulator().run(chamber, config, synthesis_method='strict_pfs')
```

Full schema + service-side integration contract: see [`doc/MIMO_First_Integration_Spec.md`](doc/MIMO_First_Integration_Spec.md).

### Dual-Polarization (Phase 6+)

Enable `ChamberConfig.dual_polarized=True` and set `AntennaArrayConfig.polarization` to `'V'` or `'H'` for each Tx/Rx antenna. The synthesizer applies TR 38.901 §7.3.2 per-cluster 2×2 polarization mixing using the cluster's `xpr_db` and `initial_phases_rad`. Slant ±45° / arbitrary polarization is Phase 7 (deferred). See `docs/api/mimo_ota_simulator.md` §8 for the behavior matrix.


### 已有簇成员表的多端口离线对照（P1-83）

对每个有成员依据的阵元对调用 `mimo_ota_simulator.joint_cluster_response.reduce_link`，按相同的成员编号归并，同时保留本链路的复系数、时延和角度统计。输入时延为秒、多普勒为 Hz、方位角和天顶角为度。对无成员依据的帧或端口保留缺失。阵元对使用自身接收复系数的功率作权重；OTA 入射场需另外计算。

对已有逐帧簇表可运行下列命令。两个容差由调用方选择，用于推定几何关联。输出包含输入校验值、各帧轨迹编号、消失的轨迹和未匹配功率占比；它不能证明物理射线身份。

```sh
python -m mimo_ota_simulator.joint_trajectory clusters_joint_tx0rx0.csv --delay-limit-ns 20 --angle-limit-deg 10
```

RT-IMT-2030 的成员表示例仅覆盖第 0 帧的 TX0～3 至 RX0。以八路传导基线的相同 101 个频点（±50 MHz）逐链路对比 MPDB，四条链路各 24 簇，宽带归一化均方误差分别为 −24.19、−24.26、−23.44、−24.09 dB。38 帧参考阵元簇表可单独用于跟踪，以示例容差得到 291 次关联。其余帧的逐射线成员关系、RX1 的成员关系以及真机播放均未由本样本验证。

### RT-IMT-2030 四发下行源帧探头 ASC（P1-84）

```sh
python -m mimo_ota_simulator.imt_probe_bundle RT聚簇样本_2_MPDB.zip examples/imt_16probe_geometry.json 新输出目录 --source-rate-hz 1
```

针对这批 2.565 GHz、四发的单频数据，以 RX0 所在位置为参考，取发射 +45° 的 Jones 向量和传播极化矩阵计算接收天线之前的电场。示例为理想水平圆环，16 探头各有 theta、phi 两个馈电口；四个发射端口分别输出 32 个文件。以指定源帧更新率每帧写一行，无帧间多普勒重采样；请保留源帧时间依据。通过按入射功率对按时延排序的路径分组压缩至最多 24 个时延抽头。此法独立于上游簇成员，不声明跨帧簇身份。

坐标以米计，相对测试区中心。各探头的方位角和天顶角以度计，双极化使用球坐标基。所有链路共用相位参考；探头按理想平面波建模，未施加实测校准。程序在拟合点求解复权重，在独立位置读取最终 ASC 重建三分量电场，对中心频率和 ±50 MHz 给出逐帧误差。场误差为各验证位置的误差能量除以目标场能量，功率比为重建场能量除以目标场能量。没有设定合格阈值。仿真区域是示例的水平面，工程构建、设备许可、垂直方向空间覆盖和真机验收另行核验。
