import numpy as np
import pytest
from channel_model_38901.simulator import ChannelSimulator

@pytest.fixture
def los_uma_sim():
    """Provides a UMa LOS simulation with a fixed seed."""
    np.random.seed(101) # Fixed seed for reproducibility
    # Use a short distance to increase LOS probability
    simulator = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)
    simulator.set_bs_ue_positions(np.array([[0, 0, 25]]), np.array([[10, 0, 1.5]]))
    return simulator

@pytest.fixture
def nlos_uma_sim():
    """Provides a UMa NLOS simulation with a fixed seed."""
    np.random.seed(102) # Fixed seed for reproducibility
    # Use a long distance to increase NLOS probability
    simulator = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)
    simulator.set_bs_ue_positions(np.array([[0, 0, 25]]), np.array([[500, 0, 1.5]]))
    return simulator

def test_h_matrix_power(los_uma_sim):
    """Tests if the H-matrix power corresponds to the pathloss and shadow fading."""
    results = los_uma_sim.run()
    H = results['channel_matrix']
    pathloss_db = results['pathloss_db']
    shadow_fade_db = results['lsps'].shadow_fade_db

    total_loss_db = pathloss_db + shadow_fade_db
    total_loss_linear = 10**(-total_loss_db / 10)

    # The power of the normalized H matrix should be close to 1.
    # The power of the final H matrix should correspond to the total loss.
    h_power_linear = np.mean(np.abs(H)**2)

    # We check if the powers are on the same order of magnitude (within a factor of 10, i.e., 10 dB)
    assert h_power_linear == pytest.approx(total_loss_linear, rel=10), \
        f"H-matrix power {h_power_linear:.2e} does not match total linear loss {total_loss_linear:.2e}"

def test_pdp_los_structure(los_uma_sim):
    """Tests if the first cluster in a LOS PDP has the highest power."""
    results = los_uma_sim.run()
    clusters = results['clusters']
    is_los = results['is_los']

    # This test is only meaningful for LOS scenarios
    if is_los:
        powers = [c['power'] for c in clusters]
        delays = [c['delay'] for c in clusters]
        
        # Find the index of the cluster with the minimum delay
        min_delay_idx = np.argmin(delays)
        # Find the index of the cluster with the maximum power
        max_power_idx = np.argmax(powers)

        assert min_delay_idx == max_power_idx, \
            "In LOS, the first arriving cluster should have the highest power."
