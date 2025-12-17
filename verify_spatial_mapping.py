
import numpy as np
import sys
import os

# Ensure we can import the simulator
sys.path.append(os.getcwd())

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig
from channel_model_38901.common.data_models import EngineCluster

def verify_spatial_mapping():
    print("--- Starting Spatial Mapping Verification ---")
    
    # 1. Setup Chamber: 16 Probes, Ring distribution
    # Probe 0 at 0 deg, Probe 4 at 90 deg, Probe 8 at 180 deg...
    num_probes = 16
    chamber = ChamberConfig(
        num_probes=num_probes,
        distribution='ring',
        radius_m=1.0,
        dual_polarized=False # Simplify to single pol for clear index mapping
    )
    
    # 2. Setup Target Channel
    # We won't use the standard generator, we will inject a synthetic cluster manually
    # to bypass random generation logic for this test.
    channel_config = TargetChannelConfig(
        model_name='UMa',
        cluster_model_name='Stochastic', 
        center_frequency_hz=3.5e9,
        tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
        rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
        ue_velocity_kmh=0.0
    )
    
    simulator = MIMO_OTA_Simulator()
    
    # 3. Create a Synthetic Cluster at AoA = 90 degrees
    # In standard math: 90 deg is +Y axis.
    # In 0-based indexing: Probe 0 is 0 deg. Probe 4 is 90 deg (4 * 360/16 = 90).
    target_aoa = 90.0
    print(f"Target Cluster AoA: {target_aoa} degrees")
    
    synthetic_cluster = EngineCluster(
        delay=0.0,
        power=1.0, # Linear power
        aoa_phi=target_aoa,
        aod_phi=0.0,
        zoa_theta=90.0,
        zod_theta=90.0,
        as_aoa=5.0, # Small angle spread to focus energy
        as_aod=5.0
    )
    
    # 4. Run Weight Calculation directly
    # This is the core logic mapping Space -> Probes
    print("Calculating probe weights...")
    weights = simulator._calculate_weights_for_cluster(synthetic_cluster, chamber)
    
    # weights shape is (Num_Effective_Probes, 1)
    # Since dual_pol is False, effective = 16.
    
    # 5. Analyze Results
    print("\n--- Probe Weight Distribution ---")
    max_w = np.max(weights)
    max_idx = np.argmax(weights)
    
    for i in range(num_probes):
        angle = i * (360.0 / num_probes)
        w = weights[i, 0]
        marker = "<-- PEAK" if i == max_idx else ""
        if w > 0.01: # Filter printing
            print(f"Probe {i:2d} (Angle {angle:5.1f}°): Weight = {w:.4f} {marker}")
            
    # 6. Conclusion
    expected_idx = int(target_aoa / (360.0 / num_probes))
    print(f"\nExpected Peak Probe: {expected_idx} (Angle {expected_idx * 360/num_probes}°)")
    print(f"Actual Peak Probe:   {max_idx} (Angle {max_idx * 360/num_probes}°)")
    
    if max_idx == expected_idx:
        print("\n✅ VERIFICATION PASSED: Mapping is mathematically consistent.")
    else:
        print("\n❌ VERIFICATION FAILED: Mapping mismatch.")

if __name__ == "__main__":
    verify_spatial_mapping()
