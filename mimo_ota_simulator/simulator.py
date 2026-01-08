import numpy as np
from typing import Dict, Literal, List
import warnings

from .data_models import ChamberConfig, TargetChannelConfig
from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.common.data_models import EngineCluster

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
                warnings.warn("Calibration vector shape mismatch. Ignoring calibration.")

        return b_matrix

    @staticmethod
    def _calculate_dut_antenna_positions(antenna_config, wavelength: float) -> np.ndarray:
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
        target_angle_rad = np.deg2rad(cluster.aoa_phi)
        target_as_rad = np.deg2rad(cluster.as_aoa)

        # --- ADAPTIVE BROADENING FIX ---
        # Ensure the Gaussian width is at least a fraction of the probe spacing.
        # Otherwise, rays falling between probes are lost (weights -> 0).
        probe_spacing_rad = 2 * np.pi / num_physical_probes
        min_as_rad = probe_spacing_rad / 1.5  # Sigma approx 2/3 of spacing ensures overlap
        
        effective_as_rad = max(target_as_rad, min_as_rad)
        # -------------------------------

        angle_diffs = np.abs(probe_angles_rad - target_angle_rad)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        candidate_indices = np.argsort(angle_diffs)[:num_candidate_probes]
        weights = np.zeros(num_physical_probes)
        
        for idx in candidate_indices:
            delta_angle = np.abs(probe_angles_rad[idx] - target_angle_rad)
            delta_angle = np.minimum(delta_angle, 2 * np.pi - delta_angle)
            
            # Use effective_as_rad for Gaussian
            weights[idx] = np.exp(-np.power(delta_angle, 2) / (2 * np.power(effective_as_rad, 2)))
            
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
            synthesis_method: Literal['pinv', 'cluster', 'ray'] = 'pinv',
            probe_phase_jitter: bool = False) -> Dict:
        num_tx_antennas = channel_config.tx_antenna.num_cols * channel_config.tx_antenna.num_rows
        num_effective_probes = chamber_config.num_probes * 2 if chamber_config.dual_polarized else chamber_config.num_probes
        probe_tx_power_contributions = np.zeros((num_effective_probes, num_tx_antennas))
        cluster_probe_mapping = {}
        ray_components = None # Initialize for export
        jitter_vector = None
        
        # Wideband Impulse Response: List of taps {delay, coeff_matrix (PxT)}
        impulse_response = []
        
        if channel_config.random_seed is not None:
            np.random.seed(channel_config.random_seed)
        
        # Quiet Zone Check
        if chamber_config.quiet_zone_radius_m is not None:
            speed_of_light = 299792458.0
            wavelength = speed_of_light / channel_config.center_frequency_hz
            dut_positions = self._calculate_dut_antenna_positions(channel_config.rx_antenna, wavelength)
            max_dut_dimension = np.max(np.linalg.norm(dut_positions, axis=1))
            if max_dut_dimension > chamber_config.quiet_zone_radius_m:
                warnings.warn(f"Warning: DUT dimension ({max_dut_dimension:.2f}m) may exceed quiet zone radius ({chamber_config.quiet_zone_radius_m:.2f}m).")

        # Synthesize the physical B-Matrix
        B_matrix = self._synthesize_b_matrix(chamber_config, channel_config)

        # --- REFACTOR: Use ChannelSimulator Facade ---
        simulator = ChannelSimulator(
            scenario_name=channel_config.model_name,
            center_frequency_hz=channel_config.center_frequency_hz,
            cluster_model_name=channel_config.cluster_model_name,
            use_median_lsps=channel_config.use_median_lsps
        )
        
        # Configure Simulator
        simulator.set_bs_ue_positions(
            bs_positions=np.array([channel_config.bs_position]),
            ue_positions=np.array([channel_config.ue_position])
        )
        simulator.set_velocities(np.array([channel_config.ue_velocity]))
        
        simulator.set_antenna_config(
            num_tx_rows=channel_config.tx_antenna.num_rows,
            num_tx_cols=channel_config.tx_antenna.num_cols,
            num_rx_rows=channel_config.rx_antenna.num_rows,
            num_rx_cols=channel_config.rx_antenna.num_cols,
            tx_spacing_h=channel_config.tx_antenna.spacing_h,
            tx_spacing_v=channel_config.tx_antenna.spacing_v,
            rx_spacing_h=channel_config.rx_antenna.spacing_h,
            rx_spacing_v=channel_config.rx_antenna.spacing_v,
            tx_array_type=channel_config.tx_antenna.array_type,
            rx_array_type=channel_config.rx_antenna.array_type
        )
        
        # Override antenna patterns if needed (Simulator facade defaults to Isotropic if not exposed, 
        # but ChannelSimulator.config.antenna_config is accessible)
        # Note: Patterns are passed to generator during init in Facade, 
        # but current AntennaArray model defaults to Isotropic anyway. 
        # If needed, update config directly for pattern strings.
        simulator.config.antenna_config.tx_element_pattern = channel_config.tx_antenna.element_pattern
        simulator.config.antenna_config.rx_element_pattern = channel_config.rx_antenna.element_pattern

        # Generate Channel
        sim_result = simulator.run(force_condition=channel_config.force_condition)
        
        target_H_matrix = sim_result['channel_matrix']
        physical_clusters = sim_result['clusters']
        lsps = sim_result['lsps']
        
        # ---------------------------------------------

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
                
                # FIX: Power is already linear from ChannelGenerator
                cluster_power_linear = cluster.power 
                
                A_cluster = np.sqrt(cluster_power_linear) * (w_map @ g_tx)
                A_total += A_cluster
                
                # Store Wideband Tap
                impulse_response.append({
                    'delay': cluster.delay,
                    'coeff': A_cluster, # (PxT)
                    'cluster_index': i
                })

                power_contributions = (w_map * cluster_power_linear).flatten()
                cluster_probe_mapping[i] = power_contributions

            transmit_mapping_matrix_A = A_total
            probe_tx_power_contributions = np.abs(transmit_mapping_matrix_A)**2
            
        elif synthesis_method == 'ray':
            # Ray-Level Synthesis (Geometric GSCM)
            # This reconstructs rays from cluster parameters, bypassing the 'Rank-1' issue of simplified models.
            
            # 1. Get Ray Offsets
            scenario_params = simulator.generator.params.get_scenario_params(simulator.config.scenario_name)
            if 'ray_angle_offsets_deg' in scenario_params:
                ray_offsets = np.array(scenario_params['ray_angle_offsets_deg'])
            else:
                ray_offsets = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 
                                        0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])
            
            A_total = np.zeros((num_effective_probes, num_tx_antennas), dtype=complex)
            c = 3e8
            lambda_0 = c / channel_config.center_frequency_hz
            k_factor = 2 * np.pi / lambda_0
            
            # Store ray components for time-series generation (Doppler)
            ray_components = []
            
            for i, cluster in enumerate(physical_clusters):
                # We will map each ray to probes independently
                cluster_power_accum = np.zeros(num_effective_probes)
                cluster_A_accum = np.zeros((num_effective_probes, num_tx_antennas), dtype=complex)
                
                for m in range(20):
                    alpha = ray_offsets[m]
                    c_scale = 1.0 # Scaling factor
                    
                    # 1. Calculate Ray Angles
                    aoa_ray = cluster.aoa_phi + c_scale * alpha * cluster.as_aoa
                    aod_ray = cluster.aod_phi + c_scale * alpha * cluster.as_aod
                    zoa_ray = cluster.zoa_theta + c_scale * alpha * cluster.as_zoa
                    zod_ray = cluster.zod_theta + c_scale * alpha * cluster.as_zod
                    
                    # 2. Calculate Probe Weights for this Ray (Point Source Mode)
                    # We treat each ray as a point source. 
                    # CRITICAL FIX: Do NOT use AS=0 (Nearest Neighbor). This causes phase quantization.
                    # Use a small AS (e.g., 2 deg) to enable Gaussian Interpolation between probes.
                    # This smooths the field and improves spatial accuracy.
                    temp_ray_cluster = EngineCluster(aod_phi=aod_ray, aoa_phi=aoa_ray, power=0, delay=0, as_aod=0, as_aoa=2.0)
                    w_map_ray = self._calculate_weights_for_cluster(temp_ray_cluster, chamber_config) # Shape (N_probes, 1)
                    
                    # 3. Calculate Tx Steering Vector (g_tx for this ray)
                    # v_tx = exp(j * k * r_tx . k_hat_departure)
                    theta_t_rad = np.deg2rad(zod_ray)
                    phi_t_rad = np.deg2rad(aod_ray)
                    k_hat_tx = np.array([
                        np.sin(theta_t_rad) * np.cos(phi_t_rad),
                        np.sin(theta_t_rad) * np.sin(phi_t_rad),
                        np.cos(theta_t_rad)
                    ])
                    
                    g_tx_ray = np.zeros((1, num_tx_antennas), dtype=complex)
                    for tx_idx in range(num_tx_antennas):
                        pos_tx = simulator.config.antenna_config.tx_positions[tx_idx]
                        phase_tx = k_factor * np.dot(k_hat_tx, pos_tx)
                        g_tx_ray[0, tx_idx] = np.exp(1j * phase_tx)
                    
                    # 4. Synthesize Contribution
                    # Random phase for this ray (Decorrelation source!)
                    phi_rnd = np.random.uniform(-np.pi, np.pi)
                    
                    # Power per ray
                    p_ray = cluster.power / 20.0
                    
                    # Base Weight (without random phase, for later reconstruction)
                    # W_base = sqrt(P) * w * g_tx
                    W_base = np.sqrt(p_ray) * (w_map_ray @ g_tx_ray)
                    
                    # Apply random phase for current snapshot
                    A_ray = W_base * np.exp(1j * phi_rnd)
                    
                    A_total += A_ray
                    cluster_A_accum += A_ray # Accumulate for tap
                    cluster_power_accum += (w_map_ray.flatten() * p_ray)
                    
                    # Calculate Doppler for this ray
                    # fd = v/lambda * cos(AoA - theta_v)
                    # UE velocity vector is (vx, vy, vz). 
                    # We need angle of arrival vector k_hat_rx.
                    # k_hat_rx points INTO the UE. 
                    # Doppler shift = (v . k_hat_rx) / lambda.
                    # k_hat_rx (spherical) from AoA/ZoA.
                    theta_r_rad = np.deg2rad(zoa_ray) # ZoA (from Zenith)
                    phi_r_rad = np.deg2rad(aoa_ray) # AoA
                    
                    k_hat_rx = np.array([
                        np.sin(theta_r_rad) * np.cos(phi_r_rad),
                        np.sin(theta_r_rad) * np.sin(phi_r_rad),
                        np.cos(theta_r_rad)
                    ])
                    
                    # UE Velocity
                    v_vec = channel_config.ue_velocity # tuple or array
                    v_dot_k = np.dot(v_vec, k_hat_rx)
                    fd = v_dot_k / lambda_0
                    
                    ray_components.append({
                        'W_base': W_base, # Complex (N_probes, N_tx)
                        'phi_init': phi_rnd,
                        'doppler_hz': fd,
                        'cluster_index': i # Link to parent cluster/tap
                    })

                cluster_probe_mapping[i] = cluster_power_accum
                
                # Store Wideband Tap (Aggregated Rays)
                impulse_response.append({
                    'delay': cluster.delay,
                    'coeff': cluster_A_accum,
                    'cluster_index': i
                })

            transmit_mapping_matrix_A = A_total
            probe_tx_power_contributions = np.abs(transmit_mapping_matrix_A)**2
            
        else:
            raise ValueError(f"Unknown synthesis method: {synthesis_method}")

        # Apply Probe Phase Jitter (Simulating hardware inconsistency)
        if probe_phase_jitter:
            # Jitter range: Full random phase [-pi, pi]
            # This simulates uncalibrated phase or large cable length variations,
            # which helps break the strong coherence of the simplified Cluster model.
            jitter_rad = np.pi 
            phase_noise = np.random.uniform(-jitter_rad, jitter_rad, size=(num_effective_probes, 1))
            transmit_mapping_matrix_A *= np.exp(1j * phase_noise)
            # Note: Jitter is applied to the TOTAL snapshot. 
            # If we reconstruct from rays, we must apply jitter there too.
            # We should store the jitter vector to apply it consistently.
            # Let's add jitter vector to return dict.
            jitter_vector = np.exp(1j * phase_noise)
        else:
            jitter_vector = None

        effective_H_matrix = B_matrix @ transmit_mapping_matrix_A
        return {
            "center_frequency_hz": channel_config.center_frequency_hz,
            "target_H_matrix": target_H_matrix,
            "transmit_mapping_matrix_A": transmit_mapping_matrix_A,
            "effective_H_matrix": effective_H_matrix,
            "B_matrix": B_matrix,
            "clusters": physical_clusters,
            "lsps": lsps, # Return LSPs for display
            "pathloss_db": sim_result['pathloss_db'],
            "is_los": sim_result['is_los'],
            "probe_tx_power_contributions": probe_tx_power_contributions,
            "cluster_probe_mapping": cluster_probe_mapping,
            "ray_components": ray_components, # Explicit variable
            "probe_phase_jitter_vector": jitter_vector,
            "channel_impulse_response": impulse_response # Wideband Taps
        }
