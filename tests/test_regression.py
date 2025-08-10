import numpy as np
import pytest
import os
from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.channel_generator import ChannelGenerator

# Discover golden files in the tests directory
GOLDEN_FILES = [f for f in os.listdir('tests') if f.startswith('golden_') and f.endswith('.npz')]

@pytest.mark.parametrize("golden_file_name", GOLDEN_FILES)
def test_regression(golden_file_name):
    """Compares a new simulation run against a saved golden file."""
    # --- 1. Parse parameters from the file name ---
    parts = golden_file_name.replace('.npz', '').split('_')
    scenario_parts = []
    for part in parts[1:]:
        if part.upper() in ['LOS', 'NLOS']:
            break
        scenario_parts.append(part)
    scenario = "-".join(scenario_parts)
    
    remaining_parts = golden_file_name.replace('.npz', '').split(f"{scenario}_")[1].split('_')
    condition = remaining_parts[0]
    mimo_config = remaining_parts[1].split('x')
    num_tx = int(mimo_config[0])
    num_rx = int(mimo_config[1])
    seed = int(remaining_parts[2])

    # --- 2. Load the golden data ---
    golden_path = os.path.join('tests', golden_file_name)
    golden_data = np.load(golden_path, allow_pickle=True)

    # --- 3. Re-run the simulation with the exact same configuration ---
    print(f"Running regression test for {scenario} {condition} {num_tx}x{num_rx} (seed={seed})...")
    np.random.seed(seed)
    
    distance = 50 if condition == 'LOS' else 500

    simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
    
    # Configure the simulator with the new data model
    ant_config = simulator.config.antenna_config
    ant_config.tx_array_type = 'ULA'
    ant_config.num_tx_rows = 1
    ant_config.num_tx_cols = num_tx
    ant_config.rx_array_type = 'ULA'
    ant_config.num_rx_rows = 1
    ant_config.num_rx_cols = num_rx

    c = 3e8
    lambda_meters = c / simulator.config.center_frequency_hz
    ant_config.tx_positions = np.array([[0, i * ant_config.tx_spacing_h * lambda_meters, 0] for i in range(num_tx)])
    ant_config.rx_positions = np.array([[0, i * ant_config.rx_spacing_h * lambda_meters, 0] for i in range(num_rx)])

    simulator.config.bs_positions = np.array([[0, 0, 25]])
    simulator.config.ue_positions = np.array([[distance, 0, 1.5]])

    simulator.generator = ChannelGenerator(simulator.config, simulator.params)

    while True:
        new_results = simulator.run()
        if (condition == 'LOS' and new_results['is_los']) or \
           (condition == 'NLOS' and not new_results['is_los']):
            break

    # --- 4. Compare every relevant piece of data ---
    np.testing.assert_allclose(
        new_results['channel_matrix'], golden_data['channel_matrix'],
        err_msg="Channel matrix H does not match the golden file."
    )

    np.testing.assert_allclose(
        new_results['pathloss_db'], golden_data['pathloss_db'][0],
        err_msg="Pathloss does not match the golden file."
    )

    assert new_results['is_los'] == golden_data['is_los'][0], \
        "LOS/NLOS condition does not match the golden file."

    # Compare clusters
    cluster_dtype = golden_data['clusters'].dtype
    new_clusters_structured = np.array([tuple(c.values()) for c in new_results['clusters']], dtype=cluster_dtype)
    assert np.all(new_clusters_structured == golden_data['clusters']), \
        "Cluster data does not match the golden file."

    # Compare LSPs
    new_lsps_dict = new_results['lsps'].model_dump()
    golden_lsps_dict = golden_data['lsps'][0]
    assert new_lsps_dict == golden_lsps_dict, \
        f"LSP data does not match the golden file. Got {new_lsps_dict}, expected {golden_lsps_dict}"

    print(f"  -> PASSED: {golden_file_name}")