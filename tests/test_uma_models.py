
import unittest
import os
import numpy as np
from channel_model_38901.parameters.parameter_loader import ParameterLoader
from channel_model_38901.models.uma_models import UMa_ChannelCondition, UMa_Pathloss, UMa_Fading
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters

class TestUMaModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load parameters once for all tests
        test_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(test_dir)
        json_path = os.path.join(
            project_root, 
            'channel_model_38901', 
            'parameters', 
            'parameters_tr38901_v19.0.0.json'
        )
        cls.param_tables = ParameterLoader.load_parameters(json_path)
        cls.uma_condition = UMa_ChannelCondition()
        cls.uma_pathloss = UMa_Pathloss(cls.param_tables)
        cls.uma_fading = UMa_Fading(cls.param_tables)

    def test_uma_condition(self):
        """Test UMa channel condition model."""
        is_los = self.uma_condition.get_condition(distance_2d=100, h_ut=1.5)
        self.assertIsInstance(is_los, bool)

    def test_uma_pathloss_los(self):
        """Test UMa pathloss model for LOS against a known value."""
        config = ScenarioConfig(
            scenario_name='UMa',
            center_frequency_hz=3.5e9,
            bs_positions=np.array([[0, 0, 25]]),
            ue_positions=np.array([[100, 0, 1.5]]),
            ue_velocities=np.array([[0, 0, 0]])
        )
        # Manual calculation for d_bp > d_2d (i.e., PL1)
        # PL1 = 28.0 + 22 * log10(100.27) + 20 * log10(3.5) = 28.0 + 22 * 2.001 + 20 * 0.544 = 82.9
        pathloss = self.uma_pathloss.calculate_pathloss(config, is_los=True)
        self.assertAlmostEqual(pathloss, 83.14, delta=0.1)

    def test_uma_fading_generation(self):
        """Test that UMa fading model generates a valid LSP set."""
        lsps = self.uma_fading.generate_lsps(is_los=True, fc_ghz=3.5)
        self.assertIsInstance(lsps, LargeScaleParameters)
        self.assertGreater(lsps.delay_spread_s, 0)
        self.assertGreater(lsps.k_factor_linear, 0)

if __name__ == '__main__':
    unittest.main()
