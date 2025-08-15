import numpy as np
import pytest
from mimo_ota_simulator.mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.mimo_ota_simulator.data_models import ChamberConfig, ProbeAntennaConfig, TargetChannelConfig, AntennaArrayConfig


class TestChamberDesign:
    def test_calculate_probe_positions_ring_4_probes(self):
        """Tests the probe position calculation for a simple 4-probe ring."""
        # 1. Setup
        simulator = MIMO_OTA_Simulator()
        chamber_config = ChamberConfig(
            num_probes=4,
            dual_polarized=False,
            distribution='ring',
            radius_m=1.0
        )

        # 2. Action
        actual_positions = simulator._calculate_probe_positions(chamber_config)

        # 3. Assertion
        expected_positions = np.array([
            [1.0, 0.0, 0.0],   # 0 degrees
            [0.0, 1.0, 0.0],   # 90 degrees
            [-1.0, 0.0, 0.0],  # 180 degrees
            [0.0, -1.0, 0.0]   # 270 degrees
        ])

        assert actual_positions.shape == (4, 3)
        np.testing.assert_allclose(actual_positions, expected_positions, atol=1e-7)

    def test_calculate_probe_positions_ring_8_probes(self):
        """Tests the probe position calculation for an 8-probe ring."""
        # 1. Setup
        simulator = MIMO_OTA_Simulator()
        chamber_config = ChamberConfig(
            num_probes=8,
            dual_polarized=False,
            distribution='ring',
            radius_m=1.0
        )

        # 2. Action
        actual_positions = simulator._calculate_probe_positions(chamber_config)

        # 3. Assertion
        # Expected angles: 0, 45, 90, 135, 180, 225, 270, 315 degrees
        sqrt2_inv = 1 / np.sqrt(2)
        expected_positions = np.array([
            [1.0, 0.0, 0.0],          # 0
            [sqrt2_inv, sqrt2_inv, 0.0],   # 45
            [0.0, 1.0, 0.0],          # 90
            [-sqrt2_inv, sqrt2_inv, 0.0],  # 135
            [-1.0, 0.0, 0.0],         # 180
            [-sqrt2_inv, -sqrt2_inv, 0.0], # 225
            [0.0, -1.0, 0.0],         # 270
            [sqrt2_inv, -sqrt2_inv, 0.0]  # 315
        ])

        assert actual_positions.shape == (8, 3)
        np.testing.assert_allclose(actual_positions, expected_positions, atol=1e-7)

    def test_chamber_config_with_probe_antenna(self):
        """Tests that ChamberConfig can be initialized with probe antenna specs."""
        # 1. Setup
        probe_antenna_config = ProbeAntennaConfig(gain_dBi=2.0, pattern='Isotropic')
        
        # 2. Action & Assertion
        chamber_config = ChamberConfig(
            num_probes=4,
            dual_polarized=False,
            distribution='ring',
            radius_m=1.0,
            probe_antenna=probe_antenna_config
        )
        
        assert chamber_config.probe_antenna.gain_dBi == 2.0

    def test_calculate_physical_channel_properties(self):
        """Tests the calculation of path loss and phase factor."""
        # 1. Setup
        simulator = MIMO_OTA_Simulator()
        probe_position = np.array([2.0, 0.0, 0.0]) # 2 meters away
        center_freq_hz = 3e9 # 3 GHz
        speed_of_light = 299792458.0

        # 2. Pre-calculation (Manual)
        distance = np.linalg.norm(probe_position)
        wavelength = speed_of_light / center_freq_hz
        # Friis formula for path loss in dB for isotropic antennas
        expected_path_loss_db = 20 * np.log10(distance) + 20 * np.log10(center_freq_hz) - 147.55
        # Phase factor e^(-j*2*pi*d/lambda)
        expected_phase_factor = np.exp(-1j * 2 * np.pi * distance / wavelength)

        # 3. Action
        path_loss_db, phase_factor = simulator._calculate_physical_channel_properties(
            probe_position, center_freq_hz
        )

        # 4. Assertion
        assert np.isclose(path_loss_db, expected_path_loss_db)
        assert np.isclose(phase_factor, expected_phase_factor)

    def test_synthesize_b_matrix_isotropic(self):
        """Tests the B-matrix synthesis for a simple isotropic case."""
        # 1. Setup
        simulator = MIMO_OTA_Simulator()
        center_freq_hz = 3e9 # 3 GHz

        chamber_config = ChamberConfig(
            num_probes=4,
            dual_polarized=False,
            distribution='ring',
            radius_m=1.0,
            probe_antenna=ProbeAntennaConfig(gain_dBi=0.0, pattern='Isotropic')
        )
        target_config = TargetChannelConfig(
            center_frequency_hz=center_freq_hz,
            rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1, element_pattern='Isotropic')
        )

        # 2. Pre-calculation (Manual)
        probe_positions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ])
        expected_B = np.zeros((1, 4), dtype=complex)
        for i in range(4):
            path_loss_db, phase_factor = simulator._calculate_physical_channel_properties(
                probe_positions[i], center_freq_hz
            )
            path_gain_linear = 10**(-path_loss_db / 20.0)
            # For isotropic antennas and no polarization loss, gain is 1
            expected_B[0, i] = path_gain_linear * phase_factor

        # 3. Action
        actual_B = simulator._synthesize_b_matrix(chamber_config, target_config)

        # 4. Assertion
        assert actual_B.shape == (1, 4)
        np.testing.assert_allclose(actual_B, expected_B, atol=1e-7)
