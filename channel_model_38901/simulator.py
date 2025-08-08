from typing import Optional
from .common.data_models import ScenarioConfig
from .parameters.parameter_loader import ParameterLoader
from .channel_generator import ChannelGenerator
import numpy as np

class ChannelSimulator:
    """
    Facade class providing a simple API for running channel simulations.
    """
    def __init__(self, scenario_name: str, center_frequency_hz: float):
        """
        Initializes the simulator for a given scenario and frequency.
        """
        # For now, we assume a fixed parameter file path
        # In a real application, this might be more configurable
        param_file = 'channel_model_38901/parameters/parameters_tr38901_v19.0.0.json'
        self.params = ParameterLoader.load_parameters(param_file)
        
        # Create a default or placeholder config
        self.config = ScenarioConfig(
            scenario_name=scenario_name,
            center_frequency_hz=center_frequency_hz,
            bs_positions=np.array([[0, 0, 25]]),
            ue_positions=np.array([[100, 0, 1.5]]),
            ue_velocities=np.array([[0, 0, 0]])
        )
        
        self.generator = ChannelGenerator(self.config, self.params)

    def run(self, force_condition: Optional[str] = None):
        """
        Runs a single channel realization and returns a dictionary of results.
        """
        print(f"Running simulation for scenario: {self.config.scenario_name}")
        H_final, clusters, lsps, pathloss_db, is_los = self.generator.generate_channel_realization(force_condition=force_condition)
        
        return {
            'channel_matrix': H_final,
            'clusters': clusters,
            'lsps': lsps,
            'pathloss_db': pathloss_db,
            'is_los': is_los
        }

    def set_bs_ue_positions(self, bs_positions: np.ndarray, ue_positions: np.ndarray):
        """Allows user to update BS and UE positions."""
        self.config.bs_positions = bs_positions
        self.config.ue_positions = ue_positions
        # Re-initialize generator with new config if necessary
        self.generator = ChannelGenerator(self.config, self.params)
