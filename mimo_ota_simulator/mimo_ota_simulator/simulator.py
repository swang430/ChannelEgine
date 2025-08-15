import numpy as np
from typing import Dict, Literal, List
import os
import channel_model_38901
import streamlit as st

from .data_models import ChamberConfig, TargetChannelConfig, ProbeAntennaConfig, AntennaArrayConfig
from channel_model_38901.common.data_models import ScenarioConfig as EngineScenarioConfig, EngineCluster, AntennaArray as EngineAntennaArray
from channel_model_38901.parameters.parameter_loader import ParameterLoader
from channel_model_38901.channel_generator import ChannelGenerator

class MIMO_OTA_Simulator:
    def _calculate_probe_positions(self, chamber_config: ChamberConfig) -> np.ndarray:
        """Calculates the 3D Cartesian coordinates of each physical probe."""
        if chamber_config.distribution == 'ring':
            num_probes = chamber_config.num_probes
            radius = chamber_config.radius_m
            angles = np.arange(num_probes) * (2 * np.pi / num_probes)
            x_coords = radius * np.cos(angles)
            y_coords = radius * np.sin(angles)
            z_coords = np.zeros(num_probes)
            return np.stack([x_coords, y_coords, z_coords], axis=1)
        else:
            raise NotImplementedError(f"Probe distribution '{chamber_config.distribution}' is not yet supported.")

    @staticmethod
    def _calculate_physical_channel_properties(probe_position: np.ndarray, center_freq_hz: float) -> tuple[float, complex]:
        """Calculates the free-space path loss and phase factor for a single link."""
        speed_of_light = 299792458.0
        distance = np.linalg.norm(probe_position)
        if distance < 1e-9:
            return 0.0, 1.0
        wavelength = speed_of_light / center_freq_hz
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(center_freq_hz) - 147.55
        phase_factor = np.exp(-1j * 2 * np.pi * distance / wavelength)
        return path_loss_db, phase_factor

    def _synthesize_b_matrix(self, chamber_config: ChamberConfig, target_config: TargetChannelConfig) -> np.ndarray:
        """Synthesizes the B-matrix based on physical properties."""
        num_rx_antennas = target_config.rx_antenna.num_rows * target_config.rx_antenna.num_cols
        num_effective_probes = chamber_config.num_probes * (2 if chamber_config.dual_polarized else 1)
        center_freq_hz = target_config.center_frequency_hz
        speed_of_light = 299792458.0
        wavelength = speed_of_light / center_freq_hz

        b_matrix = np.zeros((num_rx_antennas, num_effective_probes), dtype=complex)
        
        probe_positions = self._calculate_probe_positions(chamber_config)
        dut_antenna_positions = self._calculate_dut_antenna_positions(target_config.rx_antenna, wavelength)

        for p_idx in range(chamber_config.num_probes):
            probe_pos = probe_positions[p_idx]
            for r_idx in range(num_rx_antennas):
                rx_ant_pos = dut_antenna_positions[r_idx]
                distance_vec = probe_pos - rx_ant_pos
                
                path_loss_db, phase_factor = self._calculate_physical_channel_properties(
                    distance_vec, center_freq_hz
                )
                path_gain_linear = 10**(-path_loss_db / 20.0)

                # Incorporate probe antenna gain
                probe_gain_linear = 10**(chamber_config.probe_antenna.gain_dBi / 20.0)
                rx_gain_linear = 1.0 # Assume isotropic DUT antenna for now

                polarization_factor = 1.0

                channel_response = path_gain_linear * phase_factor * probe_gain_linear * rx_gain_linear * polarization_factor

                if not chamber_config.dual_polarized:
                    b_matrix[r_idx, p_idx] = channel_response
                else:
                    b_matrix[r_idx, 2*p_idx] = channel_response
                    b_matrix[r_idx, 2*p_idx + 1] = channel_response

        # Apply calibration vector if provided
        if chamber_config.calibration_vector is not None:
            if chamber_config.calibration_vector.shape[0] == num_effective_probes:
                b_matrix *= chamber_config.calibration_vector
            else:
                st.warning("Calibration vector shape mismatch. Ignoring calibration.")

        return b_matrix

    @staticmethod
    def _calculate_dut_antenna_positions(antenna_config: AntennaArrayConfig, wavelength: float) -> np.ndarray:
        """Calculates the 3D Cartesian coordinates of each DUT antenna element."""
        num_rows = antenna_config.num_rows
        num_cols = antenna_config.num_cols
        spacing_h = antenna_config.spacing_h * wavelength
        spacing_v = antenna_config.spacing_v * wavelength

        if antenna_config.array_type == 'ULA':
            x_coords = np.linspace(-0.5 * (num_cols - 1) * spacing_h, 0.5 * (num_cols - 1) * spacing_h, num_cols)
            y_coords = np.zeros(num_cols)
            z_coords = np.zeros(num_cols)
            return np.stack([x_coords, y_coords, z_coords], axis=1)
        
        elif antenna_config.array_type == 'URA':
            x_coords = np.linspace(-0.5 * (num_cols - 1) * spacing_h, 0.5 * (num_cols - 1) * spacing_h, num_cols)
            z_coords = np.linspace(-0.5 * (num_rows - 1) * spacing_v, 0.5 * (num_rows - 1) * spacing_v, num_rows)
            xx, zz = np.meshgrid(x_coords, z_coords)
            yy = np.zeros_like(xx)
            return np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        
        else:
            raise NotImplementedError(f"DUT antenna array type '{antenna_config.array_type}' is not yet supported.")

    @staticmethod
    def _calculate_weights_for_cluster(cluster: EngineCluster, chamber_config: ChamberConfig, num_candidate_probes: int = 4) -> np.ndarray:
        """Calculates probe weights for a single cluster to synthesize its AoA and AS."""
        num_physical_probes = chamber_config.num_probes
        num_effective_probes = num_physical_probes * 2 if chamber_config.dual_polarized else num_physical_probes
        probe_angles_rad = np.linspace(0, 2 * np.pi, num_physical_probes, endpoint=False)
        aoa_rad = np.deg2rad(cluster.aod_phi)
        as_rad = np.deg2rad(cluster.as_aod)
        angle_diffs = np.abs(probe_angles_rad - aoa_rad)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        candidate_indices = np.argsort(angle_diffs)[:num_candidate_probes]
        weights = np.zeros(num_physical_probes)
        for idx in candidate_indices:
            delta_angle = np.abs(probe_angles_rad[idx] - aoa_rad)
            delta_angle = np.minimum(delta_angle, 2 * np.pi - delta_angle)
            if as_rad > 1e-9:
                weights[idx] = np.exp(-np.power(delta_angle, 2) / (2 * np.power(as_rad, 2)))
            else:
                weights[idx] = 1.0 if np.isclose(delta_angle, 0) else 0.0
        total_weight = np.sum(weights)
        if total_weight > 1e-9:
            weights /= total_weight
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
        num_tx_antennas = channel_config.tx_antenna.num_cols * channel_config.tx_antenna.num_rows
        num_effective_probes = chamber_config.num_probes * 2 if chamber_config.dual_polarized else chamber_config.num_probes
        probe_tx_power_contributions = np.zeros((num_effective_probes, num_tx_antennas))
        cluster_probe_mapping = {}
        if channel_config.random_seed is not None:
            np.random.seed(channel_config.random_seed)
        
        # Quiet Zone Check
        if chamber_config.quiet_zone_radius_m is not None:
            speed_of_light = 299792458.0
            wavelength = speed_of_light / channel_config.center_frequency_hz
            dut_positions = self._calculate_dut_antenna_positions(channel_config.rx_antenna, wavelength)
            max_dut_dimension = np.max(np.linalg.norm(dut_positions, axis=1))
            if max_dut_dimension > chamber_config.quiet_zone_radius_m:
                st.warning(f"Warning: DUT dimension ({max_dut_dimension:.2f}m) may exceed quiet zone radius ({chamber_config.quiet_zone_radius_m:.2f}m).")

        # Synthesize the physical B-Matrix
        B_matrix = self._synthesize_b_matrix(chamber_config, channel_config)

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
            B_pinv = np.linalg.pinv(B_matrix)
            transmit_mapping_matrix_A = B_pinv @ target_H_matrix
            probe_tx_power_contributions = np.abs(transmit_mapping_matrix_A)**2

        elif synthesis_method == 'cluster':
            if not physical_clusters or physical_clusters[0].h_matrix is None:
                raise ValueError("Cluster synthesis requires h_matrix in clusters. Engine not updated?")
            
            A_total = np.zeros((num_effective_probes, num_tx_antennas), dtype=complex)

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

        effective_H_matrix = B_matrix @ transmit_mapping_matrix_A
        return {
            "target_H_matrix": target_H_matrix,
            "transmit_mapping_matrix_A": transmit_mapping_matrix_A,
            "effective_H_matrix": effective_H_matrix,
            "B_matrix": B_matrix,
            "clusters": physical_clusters,
            "probe_tx_power_contributions": probe_tx_power_contributions,
            "cluster_probe_mapping": cluster_probe_mapping,
        }
