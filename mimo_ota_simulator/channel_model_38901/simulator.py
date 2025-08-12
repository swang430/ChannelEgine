from typing import Optional, Callable, Dict
from .common.data_models import ScenarioConfig, EngineCluster, LargeScaleParameters
from .parameters.parameter_loader import ParameterLoader
from .channel_generator import ChannelGenerator
import numpy as np

class ChannelSimulator:
    """
    Facade class providing a simple API for running channel simulations.
    """
    def __init__(self, scenario_name: str, center_frequency_hz: float):
        param_file = 'channel_model_38901/parameters/parameters_tr38901_v19.0.0.json'
        self.params = ParameterLoader.load_parameters(param_file)
        
        self.config = ScenarioConfig(
            scenario_name=scenario_name,
            center_frequency_hz=center_frequency_hz,
            bs_positions=np.array([[0, 0, 25]]),
            ue_positions=np.array([[100, 0, 1.5]]),
            ue_velocities=np.array([[0, 0, 0]])
        )
        
        self.generator = ChannelGenerator(self.config, self.params)

    def run(self, force_condition: Optional[str] = None, time_s: float = 0.0, logger: Optional[Callable[[str], None]] = print) -> Dict:
        """
        Runs a single channel realization and returns a dictionary of results.
        The 'clusters' entry in the dictionary is now a list of EngineCluster objects.
        """
        if logger:
            logger(f"Running simulation for scenario: {self.config.scenario_name} at time t={time_s}s")
        
        # The generator now returns a list of rich EngineCluster objects
        H_final, H_normalized, clusters, lsps, pathloss_db, is_los = self.generator.generate_channel_realization(
            force_condition=force_condition, time_s=time_s, logger=logger
        )
        
        return {
            'channel_matrix': H_final,
            'channel_matrix_normalized': H_normalized,
            'clusters': clusters, # This is now a List[EngineCluster]
            'lsps': lsps,
            'pathloss_db': pathloss_db,
            'is_los': is_los
        }

    def set_bs_ue_positions(self, bs_positions: np.ndarray, ue_positions: np.ndarray):
        """Allows user to update BS and UE positions."""
        self.config.bs_positions = bs_positions
        self.config.ue_positions = ue_positions
        # Re-initialization of the generator is implicitly handled by config changes if needed
        # For simplicity, we can assume the generator is re-initialized before the next run if config changes.
        self.generator = ChannelGenerator(self.config, self.params)
