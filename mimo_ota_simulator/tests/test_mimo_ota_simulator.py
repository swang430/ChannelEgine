import pytest
import numpy as np
from pathlib import Path

from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig
from mimo_ota_simulator.parsers import CDL_Parser
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


class TestDataModels:
    def test_chamber_config_creation(self):
        """
        Test Case 1.1: Verifies that a ChamberConfig object can be created with valid data.
        """
        num_probes = 16
        distribution = 'ring'
        radius_m = 2.0
        calibration_matrix_B = np.ones((2, 16))

        chamber_config = ChamberConfig(
            num_probes=num_probes,
            distribution=distribution,
            radius_m=radius_m,
            calibration_matrix_B=calibration_matrix_B
        )

        assert chamber_config.num_probes == num_probes
        assert chamber_config.distribution == distribution
        assert chamber_config.radius_m == radius_m
        assert np.array_equal(chamber_config.calibration_matrix_B, calibration_matrix_B)


class TestParsers:
    def test_cdl_parser(self):
        """
        Test Case 1.2: Verifies that the CDL_Parser can correctly parse a
        well-formed CDL model file.
        """
        asset_path = Path(__file__).parent / "assets" / "cdl_a_test.ini"
        parser = CDL_Parser()
        channel_profile = parser.parse(asset_path)

        assert channel_profile is not None
        assert len(channel_profile.clusters) == 2
        cluster1 = channel_profile.clusters[0]
        assert cluster1.aod_deg == 10.0
        assert cluster1.aoa_deg == 20.0
        assert cluster1.power_dB == -3.0
        assert cluster1.delay_s == 1e-9


class TestFullSimulation:
    def test_h_effective_vs_h_target_in_ideal_chamber(self):
        """
        Test Case 3.2: Verifies that the synthesized H_effective is close
        to H_target in an ideal chamber where B is the identity matrix.
        """
        # 1. Instantiate the simulator
        simulator = MIMO_OTA_Simulator()

        # 2. Define the simulation configuration
        # NOTE: The default ChannelSimulator appears to generate a 1x1 (SISO) channel.
        # To match this, we configure an ideal chamber with Nr=1 and Np=1.
        chamber_config = ChamberConfig(
            num_probes=1,
            distribution='ring',
            radius_m=1.0,
            calibration_matrix_B=np.identity(1)  # Ideal 1-probe, 1-Rx-antenna chamber
        )

        # Use a channel model that is known to exist in the engine
        channel_config = TargetChannelConfig(
            model_name='UMa',
            mimo_config='1x1',  # Matching the engine's default
            ue_speed_kph=3.0
        )

        # 3. Run the full simulation
        try:
            results = simulator.run(
                chamber_config=chamber_config,
                channel_config=channel_config
            )
        except FileNotFoundError as e:
            pytest.fail(f"Failed to find channel engine parameter file: {e}. Make sure the channel_model_38901 directory is copied correctly.")

        # 4. Verify the results
        H_target = results['target_H_matrix']
        H_effective = results['effective_H_matrix']

        # In an ideal chamber with B=I, H_effective should be very close to H_target
        assert H_target.shape == H_effective.shape, \
            f"Shape mismatch: H_target is {H_target.shape}, H_effective is {H_effective.shape}"
        assert np.allclose(H_effective, H_target), \
            f"H_effective should be close to H_target. Difference norm: {np.linalg.norm(H_effective - H_target)}"