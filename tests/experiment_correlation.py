
import numpy as np
import itertools
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig

def run_correlation_sweep():
    print("=== Spatial Correlation Full Factorial Sweep ===")
    print("Model,Method,Probes,Spacing_Lambda,Mean_Correlation")
    
    models = ['CDL-D', 'CDL-C', 'UMa']
    methods = ['pinv', 'cluster', 'ray']
    probes_list = [8, 16, 32]
    spacings = [0.1, 0.5, 1.0]
    
    num_drops = 20 # Reduced drops for speed
    simulator = MIMO_OTA_Simulator()

    for model, method, num_probes, spacing in itertools.product(models, methods, probes_list, spacings):
        R_sum = np.zeros((2, 2), dtype=complex)
        
        # Suppress warnings during sweep
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for i in range(num_drops):
                # 1. Setup Chamber
                chamber = ChamberConfig(
                    num_probes=num_probes,
                    distribution='ring',
                    radius_m=2.0,
                    dual_polarized=False,
                    probe_antenna={"gain_dBi": 0.0, "pattern": "Isotropic"}
                )
                
                # 2. Setup Target
                # 2x1 MISO to focus on Rx correlation
                target = TargetChannelConfig(
                    model_name='UMa', # Base environment
                    cluster_model_name=model if model.startswith('CDL') else None,
                    center_frequency_hz=3.5e9,
                    random_seed=1000 + i, # Vary seed
                    tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
                    rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=spacing),
                    force_condition='NLOS' if model == 'UMa' else None 
                )
                
                # 3. Run
                try:
                    result = simulator.run(chamber, target, synthesis_method=method)
                    
                    # 4. Accumulate Correlation Matrix
                    H = result['effective_H_matrix'] # Shape (2, 1)
                    R = H @ H.conj().T
                    R_sum += R
                except Exception as e:
                    continue
        
        # Calculate Mean Correlation
        R_mean = R_sum / num_drops
        r00 = np.abs(R_mean[0,0])
        r11 = np.abs(R_mean[1,1])
        r01 = np.abs(R_mean[0,1])
        
        mean_corr = 0.0
        if r00 > 0 and r11 > 0:
            mean_corr = r01 / np.sqrt(r00 * r11)
            
        print(f"{model},{method},{num_probes},{spacing:.1f},{mean_corr:.4f}")
        sys.stdout.flush()

if __name__ == "__main__":
    run_correlation_sweep()
