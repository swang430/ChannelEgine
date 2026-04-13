import json
import os
import numpy as np

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig
from mimo_ota_simulator.exporters import PropsimASCIIExporter, PropsimExportConfig

# 1. Create a Custom JSON Profile
custom_json_path = "test_custom_profile.json"
custom_profile_data = {
    "global_params": {
        "center_frequency_hz": 2.6e9,
        "pathloss_db": 85.0,
        "ue_velocity_kph": [15.0, 0, 0]
    },
    "clusters": [
        {"delay_s": 0.0, "power_relative_linear": 0.7, "aoa_deg": 10, "aod_deg": -15, "as_aoa_deg": 5.0, "as_aod_deg": 2.0},
        {"delay_s": 2e-7, "power_relative_linear": 0.2, "aoa_deg": -45, "aod_deg": 30, "as_aoa_deg": 10.0, "as_aod_deg": 5.0},
        {"delay_s": 5e-7, "power_relative_linear": 0.1, "aoa_deg": 120, "aod_deg": -80, "as_aoa_deg": 15.0, "as_aod_deg": 10.0}
    ],
    "is_los": True
}

with open(custom_json_path, 'w') as f:
    json.dump(custom_profile_data, f, indent=4)

print(f"✅ Created custom JSON profile at {custom_json_path}")

# 2. Configure Simulator Inputs
target_config = TargetChannelConfig.parse_obj({
    "input_mode": "custom",
    "custom_profile_path": custom_json_path,
    "standard": "3gpp",  
    "model_name": "Custom",
    "center_frequency_hz": 2.6e9, 
    "bandwidth_hz": 20e6,
    "tx_antenna": {"array_type": "ULA", "num_rows": 1, "num_cols": 2, "spacing_h": 0.5},
    "rx_antenna": {"array_type": "ULA", "num_rows": 1, "num_cols": 2, "spacing_h": 0.5},
    "ue_velocity": (0, 0, 0)
})

chamber_config = ChamberConfig.parse_obj({
    "num_probes": 16,
    "dual_polarized": True, # This means 32 effective probes
    "distribution": "ring",
    "radius_m": 1.5,
    "probe_antenna": {"gain_dBi": 0.0, "pattern": "Isotropic"}
})

# 3. Run Simulation
print("🚀 Running MIMO OTA Simulation with custom JSON profile...")
simulator = MIMO_OTA_Simulator()
results = simulator.run(chamber_config, target_config, synthesis_method='ray')

print(f"✅ Simulation complete!")
print(f"   - Synthesized A_matrix shape: {results['transmit_mapping_matrix_A'].shape}")
print(f"   - Effective B_matrix shape: {results['B_matrix'].shape}")
print(f"   - Number of Ray Components: {len(results['ray_components'])}")

# 4. Run Exporter
print("\n📡 Exporting to Propsim F64 format (.asc)...")
p_config = PropsimExportConfig(
    mode="B",
    duration_s=0.01,
    sample_rate_hz=1000.0, # Just 1kHz for quick test visualization
    filename="test_custom_channel.asc"
)

try:
    exporter = PropsimASCIIExporter(p_config)
    out_files = exporter.export(results, output_path="test_custom_channel.asc", direction="dl")

    if out_files:
        print(f"✅ Exporter produced {len(out_files)} files: {out_files}")
        # Show first few lines of one file
        with open(out_files[0], 'r') as f:
            print("\n--- File Snippet (First 15 lines of port 1) ---")
            for _ in range(15):
                print(f.readline().strip())
    else:
        print("❌ Export failed!")
except Exception as e:
    print(f"❌ Error during export: {e}")
