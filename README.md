
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

