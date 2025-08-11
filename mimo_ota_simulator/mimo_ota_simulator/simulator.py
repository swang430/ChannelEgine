import numpy as np
from typing import Dict, List, Literal

# Imports from local modules
from .data_models import ChamberConfig, TargetChannelConfig

# Imports from the copied channel engine for direct interaction
from channel_model_38901.common.data_models import ScenarioConfig as EngineScenarioConfig
from channel_model_38901.common.data_models import AntennaArray as EngineAntennaArray
from channel_model_38901.parameters.parameter_loader import ParameterLoader
from channel_model_38901.channel_generator import ChannelGenerator

class MIMO_OTA_Simulator:
    """
    Core class for MIMO OTA simulation.
    """

    def run(self, chamber_config: ChamberConfig, channel_config: TargetChannelConfig, 
            synthesis_method: Literal['pinv', 'cluster'] = 'pinv') -> Dict:
        """
        Runs the full MIMO OTA simulation using the V2.0 comprehensive config.

        Args:
            chamber_config: The configuration of the OTA chamber.
            channel_config: The comprehensive V2.0 configuration for the target channel.
            synthesis_method: The algorithm to use for calculating the 'A' matrix.

        Returns:
            A dictionary containing the simulation results.
        """
        # 1. Set the random seed for reproducibility
        if channel_config.random_seed is not None:
            np.random.seed(channel_config.random_seed)

        # 2. Create the engine's AntennaArray configuration from our config
        engine_antenna_config = EngineAntennaArray(
            tx_array_type=channel_config.tx_antenna.array_type,
            tx_element_pattern=channel_config.tx_antenna.element_pattern,
            num_tx_rows=channel_config.tx_antenna.num_rows,
            num_tx_cols=channel_config.tx_antenna.num_cols,
            tx_spacing_h=channel_config.tx_antenna.spacing_h,
            tx_spacing_v=channel_config.tx_antenna.spacing_v,
            rx_array_type=channel_config.rx_antenna.array_type,
            rx_element_pattern=channel_config.rx_antenna.element_pattern,
            num_rx_rows=channel_config.rx_antenna.num_rows,
            num_rx_cols=channel_config.rx_antenna.num_cols,
            rx_spacing_h=channel_config.rx_antenna.spacing_h,
            rx_spacing_v=channel_config.rx_antenna.spacing_v,
        )

        # 3. Create the engine's full ScenarioConfig
        engine_scenario_config = EngineScenarioConfig(
            scenario_name=channel_config.model_name,
            center_frequency_hz=channel_config.center_frequency_hz,
            bs_positions=np.array([channel_config.bs_position]),
            ue_positions=np.array([channel_config.ue_position]),
            ue_velocities=np.array([channel_config.ue_velocity]),
            antenna_config=engine_antenna_config
        )

        # 4. Directly instantiate and run the core ChannelGenerator
        param_file = 'channel_model_38901/parameters/parameters_tr38901_v19.0.0.json'
        params = ParameterLoader.load_parameters(param_file)
        generator = ChannelGenerator(engine_scenario_config, params)
        
        target_H_matrix, _, clusters, _, _, _ = generator.generate_channel_realization(
            force_condition=channel_config.force_condition,
            logger=None # Suppress console output from engine
        )

        # 5. Calculate the transmit mapping matrix 'A'
        if synthesis_method == 'pinv':
            B = chamber_config.calibration_matrix_B
            B_pinv = np.linalg.pinv(B)
            transmit_mapping_matrix_A = B_pinv @ target_H_matrix
        elif synthesis_method == 'cluster':
            raise NotImplementedError("The 'cluster' synthesis method is not yet implemented.")
        else:
            raise ValueError(f"Unknown synthesis method: {synthesis_method}")

        # 6. Calculate the effective channel matrix
        effective_H_matrix = self._calculate_effective_channel(
            transmit_mapping_matrix_A=transmit_mapping_matrix_A,
            chamber_calibration_matrix_B=chamber_config.calibration_matrix_B
        )

        return {
            "target_H_matrix": target_H_matrix,
            "transmit_mapping_matrix_A": transmit_mapping_matrix_A,
            "effective_H_matrix": effective_H_matrix,
            "clusters": clusters,
            "probe_signals": {},
        }

    @staticmethod
    def _calculate_effective_channel(transmit_mapping_matrix_A: np.ndarray, chamber_calibration_matrix_B: np.ndarray) -> np.ndarray:
        """Calculates the effective channel H_effective = B @ A."""
        return chamber_calibration_matrix_B @ transmit_mapping_matrix_A