
import unittest
import os
from channel_model_38901.parameters.parameter_loader import ParameterLoader

class TestParameterLoading(unittest.TestCase):

    def setUp(self):
        # Construct the absolute path to the JSON file
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.test_dir)
        self.json_path = os.path.join(
            self.project_root, 
            'channel_model_38901', 
            'parameters', 
            'parameters1013_tr38901_v19.0.0.json'
        )

    def test_load_parameters(self):
        """Test that the parameter file can be loaded correctly."""
        self.assertTrue(os.path.exists(self.json_path), f"JSON file not found at {self.json_path}")
        
        param_tables = ParameterLoader.load_parameters(self.json_path)
        self.assertIsNotNone(param_tables)
        self.assertEqual(param_tables.data['standard_version'], "19.0.0")

    def test_get_scenario_params(self):
        """Test that we can retrieve parameters for a specific scenario."""
        param_tables = ParameterLoader.load_parameters(self.json_path)
        uma_params = param_tables.get_scenario_params('UMa')
        self.assertIsNotNone(uma_params)
        self.assertIn('pathloss_los', uma_params)
        self.assertIn('formula_1', uma_params['pathloss_los'])

if __name__ == '__main__':
    unittest.main()
