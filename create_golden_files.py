import numpy as np
import os
from channel_model_38901.simulator import ChannelSimulator

# Define the configurations for which to generate golden files
# Format: (scenario, desired_condition, seed, distance)
GOLDEN_CONFIGS = [
    ('UMa', 'LOS', 101, 50),  # Short distance for high LOS probability
    ('UMi-StreetCanyon', 'NLOS', 102, 500) # Long distance for high NLOS probability
]

def generate_golden_files():
    """Generates and saves golden reference files for regression testing."""
    print("--- Generating Golden Reference Files ---")
    
    # Ensure the tests directory exists
    if not os.path.exists('tests'):
        os.makedirs('tests')

    for scenario, condition, seed, distance in GOLDEN_CONFIGS:
        print(f"Generating file for: {scenario} {condition} (seed={seed})...")
        
        # Set the seed for reproducibility
        np.random.seed(seed)
        
        # Setup simulator
        simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
        simulator.set_bs_ue_positions(np.array([[0, 0, 25]]), np.array([[distance, 0, 1.5]]))
        
        # Run simulation, ensuring we get the desired condition
        while True:
            results = simulator.run()
            if (condition == 'LOS' and results['is_los']) or \
               (condition == 'NLOS' and not results['is_los']):
                break
            print(f"  - Got wrong condition, re-running with new seed ({seed})...")
            # If we get the wrong condition, we just re-run. The seed is fixed, 
            # so the LOS/NLOS check is deterministic. This loop is just a safeguard.

        # Define the structured dtype for clusters
        cluster_dtype = [
            ('delay', 'f8'), ('power', 'f8'),
            ('aod_phi', 'f8'), ('aoa_phi', 'f8'),
            ('zod_theta', 'f8'), ('zoa_theta', 'f8'),
            ('k_n', 'f8'), ('initial_phases', '4f8')
        ]

        # Prepare data for saving. We convert complex objects to basic types.
        save_dict = {
            'channel_matrix': results['channel_matrix'],
            # Convert list of dicts to a structured numpy array for stable saving
            'clusters': np.array([tuple(c.values()) for c in results['clusters']], dtype=cluster_dtype),
            'lsps': np.array([results['lsps'].model_dump()], dtype=object),
            'pathloss_db': np.array([results['pathloss_db']]),
            'is_los': np.array([results['is_los']])
        }

        file_path = os.path.join('tests', f'golden_{scenario}_{condition}_{seed}.npz')
        np.savez(file_path, **save_dict)
        print(f"  -> Saved golden file to: {file_path}")

    print("\n--- Golden file generation complete. ---")

if __name__ == "__main__":
    generate_golden_files()