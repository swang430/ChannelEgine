import numpy as np
import pickle

from channel_model_38901.simulator import ChannelSimulator

def generate_lsp_data(scenario='UMa', condition='NLOS', num_runs=10000):
    """Runs simulations and saves the raw LSP data to a file."""
    print(f"--- Generating LSP Data for {scenario} {condition} ---")
    print(f"Number of simulation runs: {num_runs}")

    simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
    simulator.set_bs_ue_positions(np.array([[0, 0, 25]]), np.array([[200, 0, 1.5]]))

    collected_lsps = []
    print("Starting simulation loop (progress will be printed every 10%)...")
    for i in range(num_runs):
        while True:
            np.random.seed(None)
            # Run the simulator in silent mode
            results = simulator.run(logger=None)
            is_los = results['is_los']
            if (condition == 'LOS' and is_los) or (condition == 'NLOS' and not is_los):
                collected_lsps.append(results['lsps'])
                break
        
        if (i + 1) % (num_runs // 10) == 0:
            print(f"  ... {(i + 1) * 100 / num_runs:.0f}% complete")

    output_file = f'{scenario}_{condition}_lsp_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(collected_lsps, f)
    
    print(f"\nSuccessfully generated and saved {len(collected_lsps)} LSP samples to {output_file}")

if __name__ == "__main__":
    generate_lsp_data()