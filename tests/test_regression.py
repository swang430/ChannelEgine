import numpy as np
import pytest
import os
from channel_model_38901.simulator import ChannelSimulator

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
    
    # Re-join to handle multi-word scenarios correctly
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
    simulator.set_bs_ue_positions(np.array([[0, 0, 25]]), np.array([[distance, 0, 1.5]]))
    
    ant_config = simulator.config.antenna_config
    ant_config.num_tx_antennas = num_tx
    ant_config.num_rx_antennas = num_rx
    ant_config.tx_positions = np.array([[0, i * 0.5, 0] for i in range(num_tx)])
    ant_config.rx_positions = np.array([[0, i * 0.5, 0] for i in range(num_rx)])

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

    cluster_dtype = golden_data['clusters'].dtype
    new_clusters_structured = np.array([tuple(c.values()) for c in new_results['clusters']], dtype=cluster_dtype)
    np.testing.assert_equal(
        new_clusters_structured, golden_data['clusters'],
        err_msg="Cluster data does not match the golden file."
    )

    new_lsps = new_results['lsps'].model_dump()
    golden_lsps = golden_data['lsps'][0]
    assert new_lsps == golden_lsps, \
        f"LSP data does not match the golden file. Got {new_lsps}, expected {golden_lsps}"

    print(f"  -> PASSED: {golden_file_name}")
