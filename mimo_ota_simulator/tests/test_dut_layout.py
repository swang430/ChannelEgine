import numpy as np
import pytest
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import AntennaArrayConfig

class TestDutLayout:
    def test_calculate_dut_ula_positions(self):
        """Tests the DUT antenna position calculation for a ULA."""
        # 1. Setup
        simulator = MIMO_OTA_Simulator()
        rx_antenna_config = AntennaArrayConfig(
            array_type='ULA',
            num_rows=1,
            num_cols=2,
            spacing_h=0.5
        )
        lambda_ = 0.1 # Assume wavelength of 0.1m for simplicity

        # 2. Action
        actual_positions = simulator._calculate_dut_antenna_positions(rx_antenna_config, lambda_)

        # 3. Assertion
        # Expected position for a 2-element ULA centered at origin:
        # Total width = (2-1) * 0.5 * 0.1 = 0.05m
        # Positions are at -0.025 and +0.025 on the x-axis.
        expected_positions = np.array([
            [-0.025, 0.0, 0.0],
            [0.025, 0.0, 0.0]
        ])

        assert actual_positions.shape == (2, 3)
        np.testing.assert_allclose(actual_positions, expected_positions, atol=1e-7)

    def test_calculate_dut_ura_positions(self):
        """Tests the DUT antenna position calculation for a URA."""
        # 1. Setup
        simulator = MIMO_OTA_Simulator()
        rx_antenna_config = AntennaArrayConfig(
            array_type='URA',
            num_rows=2,
            num_cols=2,
            spacing_h=0.5,
            spacing_v=0.5
        )
        lambda_ = 0.1 # Assume wavelength of 0.1m for simplicity

        # 2. Action
        actual_positions = simulator._calculate_dut_antenna_positions(rx_antenna_config, lambda_)

        # 3. Assertion
        # Expected for a 2x2 URA centered at origin:
        # x-coords: -0.025, 0.025
        # z-coords: -0.025, 0.025 (URA is in x-z plane)
        expected_positions = np.array([
            [-0.025, 0.0, -0.025],
            [0.025, 0.0, -0.025],
            [-0.025, 0.0, 0.025],
            [0.025, 0.0, 0.025]
        ])

        assert actual_positions.shape == (4, 3)
        # Sort both arrays to ensure comparison is order-independent
        sorted_actual = actual_positions[np.lexsort(actual_positions.T)]
        sorted_expected = expected_positions[np.lexsort(expected_positions.T)]
        np.testing.assert_allclose(sorted_actual, sorted_expected, atol=1e-7)