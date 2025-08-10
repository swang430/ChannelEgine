import numpy as np
import os
from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.channel_generator import ChannelGenerator

# Define the configurations for which to generate golden files
# Format: (scenario, condition, seed, distance, num_tx_cols, num_rx_cols)
# Note: We assume ULA for golden files, so rows are always 1.
GOLDEN_CONFIGS = [
    ('UMa', 'LOS', 101, 50, 2, 2),  # 2x2 MIMO UMa LOS
    ('UMi-StreetCanyon', 'NLOS', 102, 500, 4, 2) # 4x2 MIMO UMi NLOS
]

def generate_golden_files():
    """Generates and saves golden reference files for regression testing."""
    print("--- Generating Golden Reference Files ---")
    
    test_dir = 'tests'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for scenario, condition, seed, distance, num_tx, num_rx in GOLDEN_CONFIGS:
        print(f"Generating file for: {scenario} {condition} {num_tx}x{num_rx} (seed={seed})...")
        
        np.random.seed(seed)
        
        simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
        
        # Configure the simulator with the new data model
        ant_config = simulator.config.antenna_config
        ant_config.tx_array_type = 'ULA'
        ant_config.num_tx_rows = 1
        ant_config.num_tx_cols = num_tx
        ant_config.rx_array_type = 'ULA'
        ant_config.num_rx_rows = 1
        ant_config.num_rx_cols = num_rx
        
        # Manually calculate antenna positions for ULA
        c = 3e8
        lambda_meters = c / simulator.config.center_frequency_hz
        ant_config.tx_positions = np.array([[0, i * ant_config.tx_spacing_h * lambda_meters, 0] for i in range(num_tx)])
        ant_config.rx_positions = np.array([[0, i * ant_config.rx_spacing_h * lambda_meters, 0] for i in range(num_rx)])

        simulator.config.bs_positions = np.array([[0, 0, 25]])
        simulator.config.ue_positions = np.array([[distance, 0, 1.5]])

        # Re-initialize the generator with the updated config
        simulator.generator = ChannelGenerator(simulator.config, simulator.params)

        # Run until the desired LOS/NLOS condition is met
        while True:
            results = simulator.run()
            if (condition == 'LOS' and results['is_los']) or \
               (condition == 'NLOS' and not results['is_los']):
                break

        # Prepare data for saving
        cluster_dtype = [
            ('delay', 'f8'), ('power', 'f8'),
            ('aod_phi', 'f8'), ('aoa_phi', 'f8'),
            ('zod_theta', 'f8'), ('zoa_theta', 'f8'),
            ('k_n', 'f8'), ('initial_phases', '4f8')
        ]

        save_dict = {
            'channel_matrix': results['channel_matrix'],
            'clusters': np.array([tuple(c.values()) for c in results['clusters']], dtype=cluster_dtype),
            'lsps': np.array([results['lsps'].model_dump()], dtype=object),
            'pathloss_db': np.array([results['pathloss_db']]),
            'is_los': np.array([results['is_los']]),
            'num_tx': num_tx,
            'num_rx': num_rx
        }

        file_path = os.path.join(test_dir, f'golden_{scenario}_{condition}_{num_tx}x{num_rx}_{seed}.npz')
        np.savez(file_path, **save_dict)
        print(f"  -> Saved golden file to: {file_path}")

    print("\n--- Golden file generation complete. ---")

if __name__ == "__main__":
    generate_golden_files()