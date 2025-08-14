import numpy as np
from typing import Dict, Literal, List
import os
import channel_model_38901

from .data_models import ChamberConfig, TargetChannelConfig
from channel_model_38901.common.data_models import ScenarioConfig as EngineScenarioConfig, EngineCluster, AntennaArray as EngineAntennaArray
from channel_model_38901.parameters.parameter_loader import ParameterLoader
from channel_model_38901.channel_generator import ChannelGenerator

class MIMO_OTA_Simulator:
    @staticmethod
    def _calculate_probe_weights_for_aoa(aoa_deg: float, chamber_config: ChamberConfig) -> np.ndarray:
        num_physical_probes = chamber_config.num_probes
        num_effective_probes = num_physical_probes * 2 if chamber_config.dual_polarized else num_physical_probes
        probe_angles_rad = np.linspace(0, 2 * np.pi, num_physical_probes, endpoint=False)
        aoa_rad = np.deg2rad(aoa_deg)
        angle_diffs = np.abs(probe_angles_rad - aoa_rad)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        sorted_indices = np.argsort(angle_diffs)
        idx1, idx2 = sorted_indices[0], sorted_indices[1]
        p1_angle_rad, p2_angle_rad = probe_angles_rad[idx1], probe_angles_rad[idx2]
        
        # Correctly calculate angular distance, including wraparound
        total_dist = np.abs(p1_angle_rad - p2_angle_rad)
        if total_dist > np.pi:
            total_dist = 2 * np.pi - total_dist
            
        # Correctly calculate individual distances for interpolation
        dist1 = np.abs(p1_angle_rad - aoa_rad)
        if dist1 > np.pi:
            dist1 = 2 * np.pi - dist1
        dist2 = np.abs(p2_angle_rad - aoa_rad)
        if dist2 > np.pi:
            dist2 = 2 * np.pi - dist2
            
        weights = np.zeros((num_effective_probes, 1), dtype=float)
        # Handle the case where the two closest probes are on opposite sides of the wraparound point
        if not np.isclose(dist1 + dist2, total_dist):
            total_dist = 2 * np.pi - total_dist

        if np.isclose(total_dist, 0):
            w1, w2 = 1.0, 0.0
        else:
            w1 = dist2 / total_dist
            w2 = dist1 / total_dist
        if chamber_config.dual_polarized:
            weights[idx1 * 2, 0], weights[idx1 * 2 + 1, 0] = w1, w1
            weights[idx2 * 2, 0], weights[idx2 * 2 + 1, 0] = w2, w2
        else:
            weights[idx1, 0], weights[idx2, 0] = w1, w2
        return weights

    @staticmethod
    def _get_g_tx_from_h_matrix(h_matrix: np.ndarray) -> np.ndarray:
        return h_matrix[0:1, :]

    @staticmethod
    def _map_physical_to_hardware_clusters(physical_clusters: List[EngineCluster], max_hw_clusters: int = 24) -> List[EngineCluster]:
        """
        Maps physical clusters with Angle Spread (AS) to a limited number of hardware clusters.
        """
        hw_clusters = []
        for p_cluster in sorted(physical_clusters, key=lambda c: c.power, reverse=True):
            if len(hw_clusters) >= max_hw_clusters:
                break

            # For now, a simple 1-to-1 mapping. AS logic to be added.
            # In the future, this will be expanded to use multiple hw_clusters to simulate AS.
            hw_clusters.append(p_cluster)
            
        return hw_clusters

    def run(self, chamber_config: ChamberConfig, channel_config: TargetChannelConfig, 
            synthesis_method: Literal['pinv', 'cluster'] = 'pinv') -> Dict:
        if channel_config.random_seed is not None:
            np.random.seed(channel_config.random_seed)
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
        engine_scenario_config = EngineScenarioConfig(
            scenario_name=channel_config.model_name,
            cluster_model_name=channel_config.cluster_model_name,
            center_frequency_hz=channel_config.center_frequency_hz,
            bs_positions=np.array([channel_config.bs_position]),
            ue_positions=np.array([channel_config.ue_position]),
            ue_velocities=np.array([channel_config.ue_velocity]),
            antenna_config=engine_antenna_config
        )
        
        # Dynamically find the parameter file path
        lib_path = os.path.dirname(channel_model_38901.__file__)
        param_file = os.path.join(lib_path, 'parameters', 'parameters_tr38901_v19.0.0.json')
        
        params = ParameterLoader.load_parameters(param_file)
        generator = ChannelGenerator(engine_scenario_config, params, use_median_lsps=channel_config.use_median_lsps)
        target_H_matrix, _, physical_clusters, _, _, _ = generator.generate_channel_realization(
            force_condition=channel_config.force_condition, logger=None
        )
        if synthesis_method == 'pinv':
            B = chamber_config.calibration_matrix_B
            B_pinv = np.linalg.pinv(B)
            transmit_mapping_matrix_A = B_pinv @ target_H_matrix
        elif synthesis_method == 'cluster':
            if not physical_clusters or physical_clusters[0].h_matrix is None:
                raise ValueError("Cluster synthesis requires h_matrix in clusters. Engine not updated?")
            
            hardware_clusters = self._map_physical_to_hardware_clusters(physical_clusters)
            
            num_tx_antennas = channel_config.tx_antenna.num_rows * channel_config.tx_antenna.num_cols
            num_effective_probes = chamber_config.num_probes * 2 if chamber_config.dual_polarized else chamber_config.num_probes
            A_total = np.zeros((num_effective_probes, num_tx_antennas), dtype=complex)
            for cluster in hardware_clusters:
                w_map = self._calculate_probe_weights_for_aoa(cluster.aod_phi, chamber_config)
                g_tx = self._get_g_tx_from_h_matrix(cluster.h_matrix)
                cluster_power_linear = 10**(cluster.power / 10.0)
                A_cluster = np.sqrt(cluster_power_linear) * (w_map @ g_tx)
                A_total += A_cluster
            transmit_mapping_matrix_A = A_total
        else:
            raise ValueError(f"Unknown synthesis method: {synthesis_method}")
        effective_H_matrix = chamber_config.calibration_matrix_B @ transmit_mapping_matrix_A
        return {
            "target_H_matrix": target_H_matrix,
            "transmit_mapping_matrix_A": transmit_mapping_matrix_A,
            "effective_H_matrix": effective_H_matrix,
            "clusters": physical_clusters, # Return the original physical clusters for analysis
            "probe_signals": {},
        }
