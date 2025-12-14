
import numpy as np
import itertools
import sys
import os
import warnings
import csv

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig

def run_comprehensive_sweep():
    # --- Configuration ---
    frequencies = [0.7e9, 3.5e9]
    models = ['CDL-C']
    probe_counts = [8, 12, 16, 24, 32]
    methods = ['ray']
    jitter_options = [False]
    
    # Fixed parameters
    radius_m = 1.5
    rx_spacing_lambda = 0.5
    num_drops = 20 # Enough for averaging
    
    results = []

    print(f"=== Starting Frequency vs Probe Count Sweep ===")
    total_combinations = len(frequencies) * len(models) * len(probe_counts) * len(methods) * len(jitter_options)
    print(f"Total combinations: {total_combinations}")

    simulator = MIMO_OTA_Simulator()
    
    count = 0
    for freq, model, num_probes, method, jitter in itertools.product(
        frequencies, models, probe_counts, methods, jitter_options
    ):
        count += 1
        if count % 10 == 0:
            print(f"Progress: {count}/{total_combinations}")

        # Setup Configs
        wavelength = 299792458.0 / freq
        
        chamber = ChamberConfig(
            num_probes=num_probes,
            distribution='ring',
            radius_m=radius_m,
            dual_polarized=False,
            probe_antenna={"gain_dBi": 0.0, "pattern": "Isotropic"}
        )
        
        target = TargetChannelConfig(
            model_name='UMa' if model == 'UMa' else 'UMa', # Base scenario
            cluster_model_name=model if model.startswith('CDL') else None, # CDL or Stochastic
            center_frequency_hz=freq,
            random_seed=None, # Will set in loop
            tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
            rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=rx_spacing_lambda), 
            force_condition='NLOS' if model in ['CDL-C', 'UMa'] else 'LOS'
        )

        R_sum = np.zeros((2, 2), dtype=complex)
        
        # Always run multiple drops to average out random phases (Ray phases are random even in CDL)
        # This is crucial for calculating Spatial Correlation E[H H^H]. 
        # A single drop (Rank 1 H) always yields Correlation 1.0 mathematically.
        current_drops = 15 
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for i in range(current_drops):
                target.random_seed = 1000 + i
                
                try:
                    sim_res = simulator.run(
                        chamber_config=chamber, 
                        channel_config=target, 
                        synthesis_method=method,
                        probe_phase_jitter=jitter
                    )
                    
                    H = sim_res['effective_H_matrix']
                    R = H @ H.conj().T
                    R_sum += R
                except Exception as e:
                    pass
        
        # Calculate Mean Correlation
        R_mean = R_sum / current_drops
        denom = np.sqrt(np.abs(R_mean[0,0]) * np.abs(R_mean[1,1]))
        correlation = 0.0
        if denom > 0:
            correlation = np.abs(R_mean[0,1]) / denom
            
        # Store Result
        results.append({
            "Freq (GHz)": freq / 1e9,
            "Model": model,
            "Probes": num_probes,
            "Method": method,
            "Jitter": "ON" if jitter else "OFF",
            "Correlation": correlation
        })
        
    # Save to CSV
    keys = results[0].keys()
    with open("tests/freq_probe_sweep_results.csv", 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    # Print Table
    print("\n=== Frequency vs Probe Count Results ===")
    header = f"{'Freq(GHz)':<10} | {'Model':<10} | {'Probes':<6} | {'Method':<8} | {'Jitter':<6} | {'Correlation':<11}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for row in results:
        print(f"{row['Freq (GHz)']:<10.1f} | {row['Model']:<10} | {row['Probes']:<6} | {row['Method']:<8} | {row['Jitter']:<6} | {row['Correlation']:<11.4f}")

if __name__ == "__main__":
    run_comprehensive_sweep()
