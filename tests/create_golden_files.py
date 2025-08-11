import numpy as np
import os
from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.channel_generator import ChannelGenerator

GOLDEN_CONFIGS = [
    ('UMa', 'LOS', 101, 50, 2, 2),
    ('UMi-StreetCanyon', 'NLOS', 102, 500, 4, 2)
]

def generate_golden_files():
    """Generates and saves golden reference files for regression testing."""
    print("--- Generating Golden Reference Files ---")
    
    for scenario, condition, seed, distance, num_tx, num_rx in GOLDEN_CONFIGS:
        print(f"Generating file for: {scenario} {condition} {num_tx}x{num_rx} (seed={seed})...")
        
        np.random.seed(seed)
        
        simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
        
        ant_config = simulator.config.antenna_config
        ant_config.tx_array_type = 'ULA'
        ant_config.num_tx_rows = 1
        ant_config.num_tx_cols = num_tx
        ant_config.rx_array_type = 'ULA'
        ant_config.num_rx_rows = 1
        ant_config.num_rx_cols = num_rx
        
        simulator.config.bs_positions = np.array([[0, 0, 25]])
        simulator.config.ue_positions = np.array([[distance, 0, 1.5]])

        simulator.generator = ChannelGenerator(simulator.config, simulator.params)

        while True:
            results = simulator.run(logger=None)
            if (condition == 'LOS' and results['is_los']) or \
               (condition == 'NLOS' and not results['is_los']):
                break

        # Define a structured dtype for saving EngineCluster objects
        cluster_dtype = [
            ('delay', 'f8'), ('power', 'f8'),
            ('aod_phi', 'f8'), ('aoa_phi', 'f8'),
            ('zod_theta', 'f8'), ('zoa_theta', 'f8'),
            ('k_n', 'f8'),
            # Note: h_matrix is complex and its size depends on MIMO config
            ('h_matrix', 'c16', (num_rx, num_tx))
        ]

        # Convert list of EngineCluster objects to a structured numpy array
        clusters_list = results['clusters']
        structured_clusters = np.empty(len(clusters_list), dtype=cluster_dtype)
        for i, c in enumerate(clusters_list):
            structured_clusters[i]['delay'] = c.delay
            structured_clusters[i]['power'] = c.power
            structured_clusters[i]['aod_phi'] = c.aod_phi
            structured_clusters[i]['aoa_phi'] = c.aoa_phi
            structured_clusters[i]['zod_theta'] = c.zod_theta
            structured_clusters[i]['zoa_theta'] = c.zoa_theta
            structured_clusters[i]['k_n'] = c.k_n
            structured_clusters[i]['h_matrix'] = c.h_matrix

        save_dict = {
            'channel_matrix': results['channel_matrix'],
            'clusters': structured_clusters,
            'lsps': np.array([results['lsps'].model_dump()], dtype=object),
            'pathloss_db': np.array([results['pathloss_db']]),
            'is_los': np.array([results['is_los']]),
        }

        file_path = os.path.join('tests', f'golden_{scenario}_{condition}_{num_tx}x{num_rx}_{seed}.npz')
        np.savez(file_path, **save_dict)
        print(f"  -> Saved golden file to: {file_path}")

    print("\n--- Golden file generation complete. ---")

if __name__ == "__main__":
    generate_golden_files()
