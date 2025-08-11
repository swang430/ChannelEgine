import numpy as np
import pytest
import os
from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.channel_generator import ChannelGenerator

GOLDEN_FILES = [f for f in os.listdir('tests') if f.startswith('golden_') and f.endswith('.npz')]

@pytest.mark.parametrize("golden_file_name", GOLDEN_FILES)
def test_regression(golden_file_name):
    """Compares a new simulation run against a saved golden file."""
    # --- 1. Parse parameters from the file name ---
    parts = golden_file_name.replace('.npz', '').split('_')
    scenario = "-".join(parts[1:-3])
    condition = parts[-3]
    mimo_config = parts[-2].split('x')
    num_tx = int(mimo_config[0])
    num_rx = int(mimo_config[1])
    seed = int(parts[-1])

    # --- 2. Load the golden data ---
    golden_path = os.path.join('tests', golden_file_name)
    golden_data = np.load(golden_path, allow_pickle=True)

    # --- 3. Re-run the simulation with the exact same configuration ---
    np.random.seed(seed)
    simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
    
    ant_config = simulator.config.antenna_config
    ant_config.tx_array_type = 'ULA'
    ant_config.num_tx_rows = 1
    ant_config.num_tx_cols = num_tx
    ant_config.rx_array_type = 'ULA'
    ant_config.num_rx_rows = 1
    ant_config.num_rx_cols = num_rx
    
    distance = 50 if condition == 'LOS' else 500
    simulator.config.bs_positions = np.array([[0, 0, 25]])
    simulator.config.ue_positions = np.array([[distance, 0, 1.5]])
    simulator.generator = ChannelGenerator(simulator.config, simulator.params)

    while True:
        new_results = simulator.run(logger=None)
        if (condition == 'LOS' and new_results['is_los']) or \
           (condition == 'NLOS' and not new_results['is_los']):
            break

    # --- 4. Compare every relevant piece of data ---
    np.testing.assert_allclose(new_results['channel_matrix'], golden_data['channel_matrix'], err_msg="Channel matrix H does not match.")
    np.testing.assert_allclose(new_results['pathloss_db'], golden_data['pathloss_db'][0], err_msg="Pathloss does not match.")
    assert new_results['is_los'] == golden_data['is_los'][0], "LOS/NLOS condition does not match."

    # Compare LSPs
    new_lsps_dict = new_results['lsps'].model_dump()
    golden_lsps_dict = golden_data['lsps'][0]
    assert new_lsps_dict == golden_lsps_dict, "LSP data does not match."

    # Compare Clusters by converting the new list of objects to a structured array
    cluster_dtype = golden_data['clusters'].dtype
    clusters_list = new_results['clusters']
    new_structured_clusters = np.empty(len(clusters_list), dtype=cluster_dtype)
    for i, c in enumerate(clusters_list):
        new_structured_clusters[i]['delay'] = c.delay
        new_structured_clusters[i]['power'] = c.power
        new_structured_clusters[i]['aod_phi'] = c.aod_phi
        new_structured_clusters[i]['aoa_phi'] = c.aoa_phi
        new_structured_clusters[i]['zod_theta'] = c.zod_theta
        new_structured_clusters[i]['zoa_theta'] = c.zoa_theta
        new_structured_clusters[i]['k_n'] = c.k_n
        new_structured_clusters[i]['h_matrix'] = c.h_matrix

    np.testing.assert_equal(new_structured_clusters, golden_data['clusters'], err_msg="Cluster data does not match.")
