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
    def _calculate_weights_for_cluster(cluster: EngineCluster, chamber_config: ChamberConfig, num_candidate_probes: int = 4) -> np.ndarray:
        """
        Calculates probe weights for a single cluster to synthesize its AoA and AS.

        This uses a Gaussian angular power spectrum to model the Angle Spread (AS).
        It selects a number of candidate probes around the cluster's central AoA
        and assigns weights to them based on the Gaussian function.

        Args:
            cluster: The EngineCluster object to be synthesized.
            chamber_config: The configuration of the anechoic chamber.
            num_candidate_probes: The number of nearest probes to use for synthesis.

        Returns:
            A numpy array representing the power weights for all probes.
        """
        num_physical_probes = chamber_config.num_probes
        num_effective_probes = num_physical_probes * 2 if chamber_config.dual_polarized else num_physical_probes
        probe_angles_rad = np.linspace(0, 2 * np.pi, num_physical_probes, endpoint=False)
        
        aoa_rad = np.deg2rad(cluster.aod_phi) # Using aod_phi as AoA for now
        as_rad = np.deg2rad(cluster.as_aod)   # Using as_aod as AS for now

        # Find k nearest candidate probes
        angle_diffs = np.abs(probe_angles_rad - aoa_rad)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        candidate_indices = np.argsort(angle_diffs)[:num_candidate_probes]

        # Calculate weights for candidate probes using Gaussian function
        weights = np.zeros(num_physical_probes)
        for idx in candidate_indices:
            delta_angle = np.abs(probe_angles_rad[idx] - aoa_rad)
            delta_angle = np.minimum(delta_angle, 2 * np.pi - delta_angle)
            if as_rad > 1e-9: # Avoid division by zero if AS is zero
                weights[idx] = np.exp(-np.power(delta_angle, 2) / (2 * np.power(as_rad, 2)))
            else: # If AS is zero, only the single closest probe gets weight
                weights[idx] = 1.0 if np.isclose(delta_angle, 0) else 0.0

        # Normalize weights to sum to 1 (for power distribution)
        total_weight = np.sum(weights)
        if total_weight > 1e-9:
            weights /= total_weight
        
        # Expand weights for dual polarization
        final_weights = np.zeros((num_effective_probes, 1), dtype=float)
        if chamber_config.dual_polarized:
            for i in range(num_physical_probes):
                final_weights[i * 2, 0] = weights[i]
                final_weights[i * 2 + 1, 0] = weights[i]
        else:
            final_weights[:, 0] = weights
            
        return final_weights

    @staticmethod
    def _get_g_tx_from_h_matrix(h_matrix: np.ndarray) -> np.ndarray:
        """Extracts the transmit-side channel vector from a cluster's H matrix."""
        return h_matrix[0:1, :]

    def run(self, chamber_config: ChamberConfig, channel_config: TargetChannelConfig, 
            synthesis_method: Literal['pinv', 'cluster'] = 'pinv') -> Dict:
        # Initialize with empty values for non-cluster synthesis methods
        num_tx_antennas = channel_config.tx_antenna.num_cols * channel_config.tx_antenna.num_rows
        num_effective_probes = chamber_config.num_probes * 2 if chamber_config.dual_polarized else chamber_config.num_probes
        probe_tx_power_contributions = np.zeros((num_effective_probes, num_tx_antennas))
        cluster_probe_mapping = {}

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
            probe_tx_power_contributions = np.abs(transmit_mapping_matrix_A)**2

        elif synthesis_method == 'cluster':
            if not physical_clusters or physical_clusters[0].h_matrix is None:
                raise ValueError("Cluster synthesis requires h_matrix in clusters. Engine not updated?")
            
            A_total = np.zeros((num_effective_probes, num_tx_antennas), dtype=complex)

            # Loop over all physical clusters returned by the engine to calculate their contributions
            for i, cluster in enumerate(physical_clusters):
                w_map = self._calculate_weights_for_cluster(cluster, chamber_config)
                g_tx = self._get_g_tx_from_h_matrix(cluster.h_matrix)
                cluster_power_linear = 10**(cluster.power / 10.0)
                
                A_cluster = np.sqrt(cluster_power_linear) * (w_map @ g_tx)
                A_total += A_cluster

                power_contributions = (w_map * cluster_power_linear).flatten()
                cluster_probe_mapping[i] = power_contributions

            transmit_mapping_matrix_A = A_total
            probe_tx_power_contributions = np.abs(transmit_mapping_matrix_A)**2
        else:
            raise ValueError(f"Unknown synthesis method: {synthesis_method}")

        effective_H_matrix = chamber_config.calibration_matrix_B @ transmit_mapping_matrix_A
        return {
            "target_H_matrix": target_H_matrix,
            "transmit_mapping_matrix_A": transmit_mapping_matrix_A,
            "effective_H_matrix": effective_H_matrix,
            "clusters": physical_clusters, # Original physical clusters for analysis
            "probe_tx_power_contributions": probe_tx_power_contributions,
            "cluster_probe_mapping": cluster_probe_mapping,
        }