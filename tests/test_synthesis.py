import unittest
import numpy as np
import pytest

from channel_model_38901.common.geometry import get_antenna_positions
from channel_model_38901.models.spatial_correlation import get_spatial_correlation_matrix, _calculate_ula_correlation
from channel_model_38901.simulator import ChannelSimulator

class TestSynthesis(unittest.TestCase):

    def test_get_antenna_positions_ula(self):
        positions = get_antenna_positions('ULA', num_rows=1, num_cols=4, spacing_h_m=0.5, spacing_v_m=0.5)
        self.assertEqual(positions.shape, (4, 3))
        expected_positions = np.array([[0. , 0. , 0. ], [0. , 0.5, 0. ], [0. , 1. , 0. ], [0. , 1.5, 0. ]])
        np.testing.assert_allclose(positions, expected_positions)

    def test_get_antenna_positions_ura(self):
        positions = get_antenna_positions('URA', num_rows=2, num_cols=2, spacing_h_m=0.5, spacing_v_m=0.2)
        self.assertEqual(positions.shape, (4, 3))
        expected_positions = np.array([[0. , 0. , 0. ], [0. , 0.5, 0. ], [0. , 0. , 0.2], [0. , 0.5, 0.2]])
        np.testing.assert_allclose(positions, expected_positions)

    def test_ura_correlation_matrix(self):
        R_ura = get_spatial_correlation_matrix(
            array_type='URA', num_rows=2, num_cols=3, 
            spacing_v=0.5, spacing_h=0.5, 
            zenith_deg=45, azimuth_deg=30, 
            zenith_spread_deg=10, azimuth_spread_deg=20
        )
        R_h = _calculate_ula_correlation(3, 30, 20, 0.5)
        R_v = _calculate_ula_correlation(2, 45, 10, 0.5)
        R_expected = np.kron(R_v, R_h)
        self.assertEqual(R_ura.shape, (6, 6))
        np.testing.assert_allclose(R_ura, R_expected)

    def test_simulation_reproducibility(self):
        """Tests if two runs with the same seed produce identical results."""
        def run_sim_for_time(t):
            np.random.seed(42)
            sim = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)
            sim.config.antenna_config.tx_array_type = 'ULA'
            sim.config.antenna_config.num_tx_rows = 1
            sim.config.antenna_config.num_tx_cols = 1
            sim.config.antenna_config.rx_array_type = 'ULA'
            sim.config.antenna_config.num_rx_rows = 1
            sim.config.antenna_config.num_rx_cols = 1
            results = sim.run(force_condition='LOS', time_s=t, logger=None)
            return results['channel_matrix'][0, 0]

        h1 = run_sim_for_time(0.0)
        h2 = run_sim_for_time(0.0)

        self.assertEqual(h1, h2)
