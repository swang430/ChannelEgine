
import pytest
import numpy as np
from channel_model_38901.simulator import ChannelSimulator
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig

class TestComprehensiveSystem:
    """
    A comprehensive system-level test suite to verify:
    1. Core Channel Engine functionality (LSP, Ray generation).
    2. FPGA Parameter Export correctness.
    3. MIMO OTA Simulator integration and physics correctness (Power scaling).
    """

    def test_core_fpga_export_logic(self):
        """
        Verify that the FPGA parameter export generates the correct number of rays
        and valid physical parameters.
        """
        # Setup: CDL-A (23 clusters), 3.5GHz
        # 1 Tx (1x1), 1 Rx (1x1) for simplicity of counting
        sim = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9, cluster_model_name='CDL-A')
        sim.set_antenna_config(1, 1, 1, 1)
        
        # Run export
        # Note: export re-runs generation, so we expect it to work standalone
        fpga_params = sim.export_fpga_sos_params()
        
        # 1. Check Row Count
        # Expect: 23 Clusters * 20 Rays * 1 Rx * 1 Tx = 460 rows
        assert len(fpga_params) == 460, f"Expected 460 rows for CDL-A SISO, got {len(fpga_params)}"
        
        # 2. Check Data Integrity
        first_ray = fpga_params[0]
        required_keys = ["Cluster_ID", "Ray_ID", "Doppler_Hz", "Weight_Real", "Weight_Imag"]
        for key in required_keys:
            assert key in first_ray, f"Missing key {key} in FPGA export"
            
        # 3. Check Doppler Spread
        # For a given cluster, the 20 rays should have slightly different Doppler frequencies
        # (unless UE velocity is 0 or perpendicular to all rays, let's ensure non-zero velocity)
        sim.set_velocities(np.array([[10.0, 0.0, 0.0]]))
        fpga_params_moving = sim.export_fpga_sos_params()
        
        # Extract Dopplers for Cluster 0
        c0_dopplers = [p['Doppler_Hz'] for p in fpga_params_moving if p['Cluster_ID'] == 0]
        assert np.std(c0_dopplers) > 0.0, "Intra-cluster Doppler spread should be non-zero for moving UE"

    def test_mimo_ota_power_scaling_bug_fix(self):
        """
        Regression Test: Verify that MIMO OTA simulator correctly handles linear power.
        
        The Bug: 
        Previous code did `10**(cluster.power / 10.0)`.
        Since `cluster.power` is already linear (0 to 1), this operation effectively 
        flattened the power profile (values became close to 1.0).
        
        The Fix Verification:
        We simulate a CDL channel where cluster powers vary significantly.
        We check that the output probe powers also vary significantly.
        """
        ota_sim = MIMO_OTA_Simulator()
        
        # Setup OTA Chamber
        chamber_config = ChamberConfig(
            num_probes=16,
            dual_polarized=False,
            distribution='ring',
            radius_m=2.0,
            probe_antenna={"gain_dBi": 0.0, "pattern": "Isotropic"}
        )
        
        # Setup Target: CDL-A (Known high dynamic range)
        # CDL-A Cluster 1 power is 0dB (normalized approx -18dB linear total?)
        # Weakest clusters are -20dB or less relative to strongest.
        target_config = TargetChannelConfig(
            model_name='UMa',
            cluster_model_name='CDL-A',
            center_frequency_hz=3.5e9,
            bs_position=(0,0,30),
            ue_position=(100,0,1.5),
            ue_velocity=(0,0,0),
            tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
            rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1)
        )
        
        # Run Simulation with 'cluster' method (where the bug was)
        results = ota_sim.run(chamber_config, target_config, synthesis_method='cluster')
        
        probe_contributions = results['probe_tx_power_contributions'] # Shape (Num_Probes, Num_Tx)
        
        # Analysis
        # Sum power across all probes to estimate total synthesized power
        total_power = np.sum(probe_contributions)
        
        # Get Cluster-to-Probe mapping (This is directly proportional to cluster power * probe_weight)
        # cluster_probe_mapping: Dict[cluster_idx] -> [power_on_probe_0, power_on_probe_1, ...]
        cluster_mapping = results['cluster_probe_mapping']
        
        # Calculate total power per synthesized cluster
        cluster_powers_synthesized = [np.sum(p_vec) for k, p_vec in cluster_mapping.items()]
        
        # Assertions
        
        # 1. Dynamic Range Check
        # CDL-A has a dynamic range of > 20dB between strongest and weakest clusters.
        # If the bug were present (all powers ~ 1.0), dynamic range would be tiny (< 1dB).
        
        max_p = np.max(cluster_powers_synthesized)
        min_p = np.min(cluster_powers_synthesized)
        
        # Avoid log(0)
        if min_p < 1e-15: min_p = 1e-15
        
        dynamic_range_db = 10 * np.log10(max_p / min_p)
        
        print(f"\nSynthesized Dynamic Range: {dynamic_range_db:.2f} dB")
        
        # CDL-A usually has ~20-25dB range. 
        # If bug exists: 10*log10(10^0.1 / 10^0.001) ~ 10*log10(1.25/1.0) ~ 1dB.
        assert dynamic_range_db > 10.0, \
            f"Dynamic range is too low ({dynamic_range_db:.2f} dB). Likely 'Power Unit Bug' (Linear vs dB) is present."

    def test_facade_integration(self):
        """
        Verify that MIMO_OTA_Simulator effectively uses the ChannelSimulator facade 
        and returns the enriched data structures (LSP, Clusters).
        """
        ota_sim = MIMO_OTA_Simulator()
        
        chamber_config = ChamberConfig(
            num_probes=8, distribution='ring', radius_m=2.0, dual_polarized=False
        )
        target_config = TargetChannelConfig(
            model_name='RMa', # Stochastic model
            center_frequency_hz=3.5e9,
            random_seed=42,
            tx_antenna=AntennaArrayConfig(),
            rx_antenna=AntennaArrayConfig()
        )
        
        results = ota_sim.run(chamber_config, target_config, synthesis_method='pinv')
        
        # Check for LSPs
        assert 'lsps' in results
        lsps = results['lsps']
        assert lsps.delay_spread_s > 0
        assert lsps.shadow_fade_db != 0 # Random value
        
        # Check for Clusters
        assert 'clusters' in results
        assert len(results['clusters']) > 0
