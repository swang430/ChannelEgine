
import numpy as np
import os
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig

def create_golden_file():
    """
    Runs a simulation with a fixed configuration and saves the output
    A-matrix to a 'golden file' for regression testing.
    """
    print("--- Generating Golden File for Regression Test ---")

    # 1. Define a fixed, deterministic configuration
    chamber_config = ChamberConfig(
        num_probes=8,
        dual_polarized=False,
        distribution='ring',
        radius_m=1.0
    )
    
    channel_config = TargetChannelConfig(
        model_name='UMi-StreetCanyon',
        cluster_model_name='CDL-A',
        center_frequency_hz=3.5e9,
        tx_antenna=AntennaArrayConfig(num_cols=4, array_type='ULA'),
        rx_antenna=AntennaArrayConfig(num_cols=2, array_type='ULA'),
        random_seed=42  # CRITICAL: Use a fixed seed for reproducibility
    )

    # 2. Run the simulation
    simulator = MIMO_OTA_Simulator()
    # Use cluster synthesis as it's more complex and a better regression target
    results = simulator.run(chamber_config, channel_config, synthesis_method='cluster')
    golden_A_matrix = results['transmit_mapping_matrix_A']

    # 3. Save the result to a file
    # Ensure the output path is inside the tests directory
    output_dir = 'tests'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'golden_results.npz')
    
    np.savez_compressed(file_path, golden_A=golden_A_matrix)

    print(f"Successfully created golden file: {file_path}")
    print(f"Golden A-matrix shape: {golden_A_matrix.shape}")
    print(f"Golden A-matrix sample value: {golden_A_matrix[0, 0]}")

if __name__ == "__main__":
    create_golden_file()
