import unittest
import os
import numpy as np
from channel_model_38901.parameters.parameter_loader import ParameterLoader
from channel_model_38901.common.data_models import ScenarioConfig
from channel_model_38901.channel_generator import ChannelGenerator

class TestChannelGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(test_dir)
        json_path = os.path.join(
            project_root, 
            'channel_model_38901', 
            'parameters', 
            'parameters_tr38901_v19.0.0.json'
        )
        cls.param_tables = ParameterLoader.load_parameters(json_path)
        cls.config = ScenarioConfig(
            scenario_name='UMa',
            center_frequency_hz=3.5e9,
            bs_positions=np.array([[0, 0, 25]]),
            ue_positions=np.array([[100, 0, 1.5]]),
            ue_velocities=np.array([[0, 0, 0]])
        )

    def test_channel_generator_execution(self):
        """Test that the channel generator runs without critical errors."""
        generator = ChannelGenerator(self.config, self.param_tables)
        results = generator.generate_channel_realization()
        channel_matrix = results[0]
        
        self.assertIsNotNone(channel_matrix)
        self.assertEqual(channel_matrix.shape, (1, 1))
        self.assertIsInstance(channel_matrix[0, 0], np.complex128)

if __name__ == '__main__':
    unittest.main()