
import pytest
import numpy as np
import os
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from channel_model_38901.common.data_models import EngineCluster

class TestCoreAlgorithms:
    @pytest.fixture
    def simulator(self):
        return MIMO_OTA_Simulator()

    def test_calculate_weights_for_cluster(self, simulator):
        """
        Tests the new weight calculation based on Gaussian distribution for Angle Spread.
        """
        chamber_config = ChamberConfig(num_probes=16, dual_polarized=False, distribution='ring', radius_m=1, calibration_matrix_B=np.ones((1,16)))
        # A cluster with a clear direction and some spread
        cluster = EngineCluster(aod_phi=45.0, as_aod=10.0) # AoA=45 deg, AS=10 deg

        weights = simulator._calculate_weights_for_cluster(cluster, chamber_config, num_candidate_probes=4)
        
        # Probe 2 is at 360/16 * 2 = 45 degrees, so it should have the highest weight
        probe_2_weight = weights[2, 0]
        # Probes 1 and 3 are neighbors
        probe_1_weight = weights[1, 0]
        probe_3_weight = weights[3, 0]
        
        assert weights.shape == (16, 1)
        assert np.isclose(np.sum(weights), 1.0) # Weights should be normalized
        assert probe_2_weight > probe_1_weight
        assert probe_2_weight > probe_3_weight
        assert np.count_nonzero(weights) == 4 # Exactly 4 probes should have weight

    def test_get_g_tx_from_h_matrix(self, simulator):
        h_matrix = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
        g_tx = simulator._get_g_tx_from_h_matrix(h_matrix)
        assert g_tx.shape == (1, 3)
        assert np.array_equal(g_tx, np.array([[1+1j, 2+2j, 3+3j]]))

class TestFullSimulation:
    @pytest.fixture
    def default_configs(self):
        chamber_config = ChamberConfig(
            num_probes=16,
            dual_polarized=True,
            distribution='ring',
            radius_m=1.5,
            calibration_matrix_B=np.ones((2, 32))
        )
        channel_config = TargetChannelConfig(
            tx_antenna=AntennaArrayConfig(num_cols=4),
            rx_antenna=AntennaArrayConfig(num_cols=2)
        )
        return chamber_config, channel_config

    def test_pinv_synthesis_and_config_passing(self, default_configs, mocker):
        """
        Enhanced Sanity Check for the 'pinv' method.
        1. Mocks the ChannelGenerator class.
        2. Verifies that the simulator initializes the generator with the correct config.
        3. Verifies the output matrix shape.
        """
        chamber_config, channel_config = default_configs

        # Mock the entire ChannelGenerator class to inspect its initialization
        mock_generator_class = mocker.patch('mimo_ota_simulator.simulator.ChannelGenerator')
        
        # Configure the MOCKED INSTANCE that will be created
        mock_instance = mock_generator_class.return_value
        mock_instance.generate_channel_realization.return_value = (
            np.ones((2, 4)), None, [], None, None, None
        )

        # Run the simulation
        simulator = MIMO_OTA_Simulator()
        results = simulator.run(chamber_config, channel_config, synthesis_method='pinv')
        
        # --- Assertions ---
        # 1. Verify the output shape (as before)
        A = results['transmit_mapping_matrix_A']
        assert A.shape == (32, 4)

        # 2. Verify that the ChannelGenerator was called correctly
        mock_generator_class.assert_called_once()
        args, _ = mock_generator_class.call_args
        
        # 3. Inspect the EngineScenarioConfig passed to the generator
        engine_config_arg = args[0]
        assert engine_config_arg.scenario_name == channel_config.model_name
        assert engine_config_arg.center_frequency_hz == channel_config.center_frequency_hz
        assert engine_config_arg.antenna_config.num_tx_cols == channel_config.tx_antenna.num_cols
        assert engine_config_arg.antenna_config.num_rx_rows == channel_config.rx_antenna.num_rows

    def test_cluster_synthesis(self, default_configs, mocker):
        chamber_config, channel_config = default_configs
        # Add as_aod to the mock clusters
        mock_clusters = [
            EngineCluster(aoa_phi=45, as_aod=5.0, power=1.0, h_matrix=np.ones((2, 4), dtype=complex)),
            EngineCluster(aoa_phi=90, as_aod=8.0, power=1.0, h_matrix=np.ones((2, 4), dtype=complex) * 2)
        ]
        mocker.patch('mimo_ota_simulator.simulator.ChannelGenerator.generate_channel_realization', return_value=(np.ones((2, 4)), None, mock_clusters, None, None, None))
        simulator = MIMO_OTA_Simulator()
        results = simulator.run(chamber_config, channel_config, synthesis_method='cluster')
        A = results['transmit_mapping_matrix_A']
        assert A.shape == (32, 4)
        assert np.iscomplexobj(A)

def test_regression_against_golden_file():
    """
    Regression Test: Compares the current output against a pre-computed 'golden' file.
    """
    # 1. Load the golden results
    golden_file_path = 'tests/golden_results.npz'
    assert os.path.exists(golden_file_path), "Golden file not found. Please generate it first."
    golden_data = np.load(golden_file_path)
    golden_A = golden_data['golden_A']

    # 2. Define the exact same configuration used to generate the golden file
    chamber_config = ChamberConfig(
        num_probes=8,
        dual_polarized=False,
        distribution='ring',
        radius_m=1.0,
        calibration_matrix_B=np.eye(2, 8)
    )
    channel_config = TargetChannelConfig(
        model_name='UMi-StreetCanyon',
        cluster_model_name='CDL-A',
        center_frequency_hz=3.5e9,
        tx_antenna=AntennaArrayConfig(num_cols=4, array_type='ULA'),
        rx_antenna=AntennaArrayConfig(num_cols=2, array_type='ULA'),
        random_seed=42
    )

    # 3. Run the simulation with the current code
    simulator = MIMO_OTA_Simulator()
    results = simulator.run(chamber_config, channel_config, synthesis_method='cluster')
    current_A = results['transmit_mapping_matrix_A']

    # 4. Assert that the current output matches the golden file
    np.testing.assert_allclose(current_A, golden_A, rtol=1e-6, atol=1e-6)
