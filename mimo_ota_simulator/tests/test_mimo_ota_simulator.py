import pytest
import numpy as np
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from channel_model_38901.common.data_models import EngineCluster

class TestCoreAlgorithms:
    @pytest.fixture
    def simulator(self):
        return MIMO_OTA_Simulator()

    @pytest.mark.parametrize("aoa, expected_idx1, expected_w1, expected_idx2, expected_w2", [
        (45, 2, 1.0, 3, 0.0),          # Exactly on probe 2 (angle 45)
        (50, 2, 0.777, 3, 0.222),     # Between probe 2 (45) and 3 (67.5)
        (355, 0, 0.777, 15, 0.222)    # Wraparound: AOA is closer to probe 0, so it gets the higher weight
    ])
    def test_calculate_probe_weights_for_aoa(self, simulator, aoa, expected_idx1, expected_w1, expected_idx2, expected_w2):
        chamber_config = ChamberConfig(num_probes=16, dual_polarized=False, distribution='ring', radius_m=1, calibration_matrix_B=np.ones((1,16)))
        weights = simulator._calculate_probe_weights_for_aoa(aoa, chamber_config)
        assert weights.shape == (16, 1)
        assert np.isclose(weights[expected_idx1, 0], expected_w1, atol=1e-3)
        assert np.isclose(weights[expected_idx2, 0], expected_w2, atol=1e-3)
        assert np.isclose(np.sum(weights), 1.0)

    def test_get_g_tx_from_h_matrix(self, simulator):
        h_matrix = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
        g_tx = simulator._get_g_tx_from_h_matrix(h_matrix)
        assert g_tx.shape == (1, 3)
        assert np.array_equal(g_tx, np.array([[1+1j, 2+2j, 3+3j]]))

    def test_map_physical_to_hardware_clusters(self, simulator):
        """
        Tests the mapping of physical clusters to hardware clusters, ensuring power-based sorting.
        """
        physical_clusters = [
            EngineCluster(power=-10.0), # Low power
            EngineCluster(power=0.0),   # High power
            EngineCluster(power=-5.0)   # Medium power
        ]
        hw_clusters = simulator._map_physical_to_hardware_clusters(physical_clusters, max_hw_clusters=2)
        
        assert len(hw_clusters) == 2
        # Verify that the two most powerful clusters were selected
        assert hw_clusters[0].power == 0.0
        assert hw_clusters[1].power == -5.0

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

    def test_pinv_synthesis(self, default_configs, mocker):
        chamber_config, channel_config = default_configs
        mocker.patch('mimo_ota_simulator.simulator.ChannelGenerator.generate_channel_realization', return_value=(np.ones((2, 4)), None, [], None, None, None))
        simulator = MIMO_OTA_Simulator()
        results = simulator.run(chamber_config, channel_config, synthesis_method='pinv')
        A = results['transmit_mapping_matrix_A']
        assert A.shape == (32, 4)

    def test_cluster_synthesis(self, default_configs, mocker):
        chamber_config, channel_config = default_configs
        mock_clusters = [
            EngineCluster(aoa_phi=45, power=1.0, h_matrix=np.ones((2, 4), dtype=complex)),
            EngineCluster(aoa_phi=90, power=1.0, h_matrix=np.ones((2, 4), dtype=complex) * 2)
        ]
        mocker.patch('mimo_ota_simulator.simulator.ChannelGenerator.generate_channel_realization', return_value=(np.ones((2, 4)), None, mock_clusters, None, None, None))
        simulator = MIMO_OTA_Simulator()
        results = simulator.run(chamber_config, channel_config, synthesis_method='cluster')
        A = results['transmit_mapping_matrix_A']
        assert A.shape == (32, 4)
        assert np.iscomplexobj(A)