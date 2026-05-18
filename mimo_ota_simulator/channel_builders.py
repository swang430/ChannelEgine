import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from mimo_ota_simulator.data_models import TargetChannelConfig, UniversalChannelProfile
from channel_model_38901.common.data_models import EngineCluster
from channel_model_38901.simulator import ChannelSimulator

class ChannelProfileBuilder(ABC):
    """Abstract base class for channel profile builders."""
    @abstractmethod
    def build(self) -> UniversalChannelProfile:
        pass

class Standard3GPPBuilder(ChannelProfileBuilder):
    """Builds a channel profile using the internal 3GPP TR 38.901 Generation Engine."""
    def __init__(self, config: TargetChannelConfig):
        self.config = config

    def build(self) -> UniversalChannelProfile:
        # Initialize the underlying 3GPP channel simulator
        simulator = ChannelSimulator(
            scenario_name=self.config.model_name,
            center_frequency_hz=self.config.center_frequency_hz,
            cluster_model_name=self.config.cluster_model_name,
            use_median_lsps=self.config.use_median_lsps
        )
        
        # Set positions and velocity
        simulator.set_bs_ue_positions(
            bs_positions=np.array([self.config.bs_position]),
            ue_positions=np.array([self.config.ue_position])
        )
        simulator.set_velocities(np.array([self.config.ue_velocity]))

        # Antenna config must flow through to the 3GPP generator so cluster.h_matrix
        # carries the correct (num_rx, num_tx) shape with per-antenna phase variation.
        # Historical bug: this section was originally skipped on the assumption that
        # MIMO_OTA_Simulator computes its own probe/antenna synthesis. That holds for
        # 'ray'/'strict_pfs'/'pinv' which derive Tx steering locally, but breaks the
        # 'cluster' method which reads g_tx = cluster.h_matrix[0:1, :]. Without this
        # call the default (1, 1) antenna config produced a rank-1 h_matrix and the
        # cluster mode emitted A with all Tx columns identical.
        tx = self.config.tx_antenna
        rx = self.config.rx_antenna
        simulator.set_antenna_config(
            num_tx_rows=tx.num_rows,
            num_tx_cols=tx.num_cols,
            num_rx_rows=rx.num_rows,
            num_rx_cols=rx.num_cols,
            tx_spacing_h=tx.spacing_h,
            tx_spacing_v=tx.spacing_v,
            rx_spacing_h=rx.spacing_h,
            rx_spacing_v=rx.spacing_v,
            tx_array_type=tx.array_type,
            rx_array_type=rx.array_type,
        )

        # Generate the channel realization
        results = simulator.run(
            force_condition=self.config.force_condition,
            use_median_lsps=self.config.use_median_lsps,
            logger=None
        )
        
        return UniversalChannelProfile(
            center_frequency_hz=self.config.center_frequency_hz,
            ue_velocity=np.array(self.config.ue_velocity),
            pathloss_db=results['pathloss_db'],
            clusters=results['clusters'],
            lsps=results['lsps'],
            is_los=results['is_los']
        )

class CustomJSONBuilder(ChannelProfileBuilder):
    """Builds a channel profile by loading explicit clusters from a custom JSON file."""
    def __init__(self, json_filepath: str):
        self.json_filepath = json_filepath

    def build(self) -> UniversalChannelProfile:
        try:
            with open(self.json_filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read custom channel profile {self.json_filepath}: {str(e)}")
            
        global_params = data.get('global_params', {})
        clusters_data = data.get('clusters', [])
        
        center_frequency_hz = global_params.get('center_frequency_hz', 2.4e9)
        pathloss_db = global_params.get('pathloss_db', 100.0)
        vel = global_params.get('ue_velocity_kph', [0.0, 0.0, 0.0])
        # Convert kph to m/s
        ue_velocity = np.array(vel) * (1000.0 / 3600.0)
        
        clusters = []
        for c in clusters_data:
            clusters.append(EngineCluster(
                delay=c.get('delay_s', 0.0),
                power=c.get('power_relative_linear', 0.0),
                aod_phi=c.get('aod_deg', 0.0),
                aoa_phi=c.get('aoa_deg', 0.0),
                zod_theta=c.get('zod_deg', 90.0),
                zoa_theta=c.get('zoa_deg', 90.0),
                as_aod=c.get('as_aod_deg', 0.0),
                as_aoa=c.get('as_aoa_deg', 0.0),
                as_zod=c.get('as_zod_deg', 0.0),
                as_zoa=c.get('as_zoa_deg', 0.0),
                k_n=1.0,
                # Random initial phases to avoid deterministic coherent additions across all tests
                initial_phases=np.random.uniform(-np.pi, np.pi, size=(4,))
            ))
            
        # Normalize powers to sum to 1.0 (linear) to be safe
        total_power = sum(c.power for c in clusters)
        if total_power > 0:
            for c in clusters:
                c.power /= total_power
                
        return UniversalChannelProfile(
            center_frequency_hz=center_frequency_hz,
            ue_velocity=ue_velocity,
            pathloss_db=pathloss_db,
            clusters=clusters,
            lsps=None,
            is_los=data.get('is_los', False)
        )

def create_builder(config: TargetChannelConfig) -> ChannelProfileBuilder:
    """Factory method to create the appropriate channel builder based on input config."""
    if config.input_mode == 'custom' and config.custom_profile_path:
        return CustomJSONBuilder(config.custom_profile_path)
    # Default to 3GPP 
    return Standard3GPPBuilder(config)
