import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig

def diagnose():
    print("=== Diagnosing Doppler Generation ===")
    
    # 1. High Velocity to guarantee Doppler
    velocity_kmh = 100.0 
    velocity_mps = velocity_kmh / 3.6
    print(f"Velocity: {velocity_kmh} km/h ({velocity_mps:.2f} m/s)")
    
    target = TargetChannelConfig(
        model_name='UMa',
        cluster_model_name='CDL-C',
        center_frequency_hz=3.5e9,
        ue_velocity=(velocity_mps, 0.0, 0.0), # X-axis motion
        tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
        rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
        force_condition='NLOS'
    )
    
    chamber = ChamberConfig(
        num_probes=16,
        distribution='ring',
        radius_m=1.5,
        probe_antenna={"gain_dBi": 0.0, "pattern": "Isotropic"}
    )
    
    sim = MIMO_OTA_Simulator()
    result = sim.run(chamber, target, synthesis_method='ray')
    
    # 2. Check Ray Components
    rays = result.get('ray_components')
    if not rays:
        print("ERROR: No ray_components returned!")
        return
        
    print(f"Total Rays: {len(rays)}")
    
    # 3. Analyze Doppler Frequencies
    dopplers = [r['doppler_hz'] for r in rays]
    max_doppler = np.max(np.abs(dopplers))
    avg_doppler = np.mean(np.abs(dopplers))
    
    print(f"Max Doppler: {max_doppler:.4f} Hz")
    print(f"Avg |Doppler|: {avg_doppler:.4f} Hz")
    
    expected_max_fd = velocity_mps / (3e8/3.5e9)
    print(f"Theoretical Max Fd: {expected_max_fd:.4f} Hz")
    
    if max_doppler < 1e-9:
        print("FAIL: Doppler is ZERO. Simulator Logic Error.")
    else:
        print("PASS: Doppler frequencies are generated.")
        
        # 4. Check Exporter Logic manually
        print("\n--- Checking Exporter Logic ---")
        from mimo_ota_simulator.exporters import PropsimASCIIExporter
        from mimo_ota_simulator.data_models import PropsimExportConfig
        
        exp_config = PropsimExportConfig(
            filename="temp_diag.asc",
            mode='B',
            duration_s=0.01,
            sample_rate_hz=1000
        )
        exporter = PropsimASCIIExporter(exp_config)
        
        # We simulate the _export_wideband_dynamic logic for 2 steps
        # Just check if the phase rotation matrix is not all 1s
        
        # Omegas = (N_rays, 1)
        Omegas = np.array(dopplers)[:, np.newaxis]
        t = 0.005 # 5ms
        phase_rot = np.exp(1j * 2 * np.pi * Omegas * t)
        
        # Check deviation from 1.0
        deviation = np.max(np.abs(phase_rot - 1.0))
        print(f"Phase Rotation at t=5ms (Max Deviation from 1.0): {deviation:.6f}")
        
        if deviation < 1e-6:
             print("FAIL: Phase rotation is negligible!")
        else:
             print("PASS: Phase rotation is significant.")

if __name__ == "__main__":
    diagnose()
