import numpy as np
import pytest
from channel_model_38901.simulator import ChannelSimulator

def test_cdl_a_channel_generation():
    """Tests that a CDL-A channel can be generated without errors."""
    # 1. Instantiate the simulator with the CDL-A scenario
    try:
        simulator = ChannelSimulator(scenario_name='CDL-A', center_frequency_hz=3.5e9)
    except Exception as e:
        pytest.fail(f"Failed to instantiate ChannelSimulator for CDL-A: {e}")

    # 2. Configure antenna dimensions based on the CDL-A model definition
    cdl_a_params = simulator.params.get_cdl_model_params('CDL-A')
    num_tx = cdl_a_params['num_tx_antennas']
    num_rx = cdl_a_params['num_rx_antennas']
    
    simulator.config.antenna_config.num_tx_cols = num_tx
    simulator.config.antenna_config.num_rx_cols = num_rx
    simulator.generator = simulator.generator.__class__(simulator.config, simulator.params)

    # 3. Run the simulation
    results = simulator.run(logger=None)

    # 4. Assertions
    assert results is not None, "Simulation results should not be None"
    
    # Check channel matrix
    h_matrix = results['channel_matrix']
    assert isinstance(h_matrix, np.ndarray), "Channel matrix should be a numpy array"
    assert h_matrix.shape == (num_rx, num_tx), f"Channel matrix shape should be {(num_rx, num_tx)}"
    
    # Check condition
    assert results['is_los'] is False, "CDL-A condition should be NLOS (False)"

    # Check clusters
    clusters = results['clusters']
    assert isinstance(clusters, list), "Clusters should be a list"
    assert len(clusters) > 0, "Should have at least one cluster"
    assert abs(clusters[0].delay - 0.0) < 1e-12, "First cluster delay should be 0.0"
    assert abs(sum(c.power for c in clusters) - 1.0) < 1e-9, "Normalized cluster powers should sum to 1.0"
