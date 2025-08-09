import numpy as np
import os
from channel_model_38901.simulator import ChannelSimulator

# Define the configurations for which to generate golden files
# Format: (scenario, condition, seed, distance, num_tx, num_rx)
GOLDEN_CONFIGS = [
    ('UMa', 'LOS', 101, 50, 2, 2),  # 2x2 MIMO UMa LOS
    ('UMi-StreetCanyon', 'NLOS', 102, 500, 4, 2) # 4x2 MIMO UMi NLOS
]

def generate_golden_files():
    """Generates and saves golden reference files for regression testing."""
    print("--- Generating Golden Reference Files ---")
    
    if not os.path.exists('tests'):
        os.makedirs('tests')

    for scenario, condition, seed, distance, num_tx, num_rx in GOLDEN_CONFIGS:
        print(f"Generating file for: {scenario} {condition} {num_tx}x{num_rx} (seed={seed})...")
        
        np.random.seed(seed)
        
        simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
        simulator.set_bs_ue_positions(np.array([[0, 0, 25]]), np.array([[distance, 0, 1.5]]))
        
        ant_config = simulator.config.antenna_config
        ant_config.num_tx_antennas = num_tx
        ant_config.num_rx_antennas = num_rx
        ant_config.tx_positions = np.array([[0, i * 0.5, 0] for i in range(num_tx)])
        ant_config.rx_positions = np.array([[0, i * 0.5, 0] for i in range(num_rx)])

        while True:
            results = simulator.run()
            if (condition == 'LOS' and results['is_los']) or \
               (condition == 'NLOS' and not results['is_los']):
                break

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

        file_path = os.path.join('tests', f'golden_{scenario}_{condition}_{num_tx}x{num_rx}_{seed}.npz')
        np.savez(file_path, **save_dict)
        print(f"  -> Saved golden file to: {file_path}")

    print("\n--- Golden file generation complete. ---")

if __name__ == "__main__":
    generate_golden_files()
