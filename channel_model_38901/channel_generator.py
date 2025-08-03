import numpy as np
from .common.data_models import ScenarioConfig, LargeScaleParameters
from .parameters.parameter_loader import ParameterTables
from .models.abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from .models.uma_models import UMa_ChannelCondition, UMa_Pathloss, UMa_Fading

from .models.umi_models import UMi_ChannelCondition, UMi_Pathloss, UMi_Fading

from .models.rma_models import RMa_ChannelCondition, RMa_Pathloss, RMa_Fading

from .models.inh_models import InH_ChannelCondition, InH_Pathloss, InH_Fading

from .models.sma_models import SMa_ChannelCondition, SMa_Pathloss, SMa_Fading

class ChannelGenerator:
    """
    Core simulation engine that generates channel realizations based on TR 38.901.
    """
    def __init__(self, config: ScenarioConfig, params: ParameterTables):
        self.config = config
        self.params = params
        self._select_models()

    def _select_models(self):
        """Selects the appropriate model implementations based on the scenario."""
        scenario = self.config.scenario_name
        if scenario == 'UMa':
            self.condition_model = UMa_ChannelCondition()
            self.pathloss_model = UMa_Pathloss(self.params)
            self.fading_model = UMa_Fading(self.params)
        elif scenario == 'UMi-StreetCanyon':
            self.condition_model = UMi_ChannelCondition()
            self.pathloss_model = UMi_Pathloss(self.params)
            self.fading_model = UMi_Fading(self.params)
        elif scenario == 'RMa':
            self.condition_model = RMa_ChannelCondition()
            self.pathloss_model = RMa_Pathloss(self.params)
            self.fading_model = RMa_Fading(self.params)
        elif scenario == 'InH-Office':
            self.condition_model = InH_ChannelCondition()
            self.pathloss_model = InH_Pathloss(self.params)
            self.fading_model = InH_Fading(self.params)
        elif scenario == 'SMa':
            self.condition_model = SMa_ChannelCondition()
            self.pathloss_model = SMa_Pathloss(self.params)
            self.fading_model = SMa_Fading(self.params)
        # TODO: Add other scenarios like UMi, RMa, etc.
        else:
            raise NotImplementedError(f"Scenario '{scenario}' is not yet supported.")

    def generate_channel_realization(self):
        """
        Executes the 12-step channel generation procedure from TR 38.901 Section 7.5.
        """
        # This will be a placeholder for the full 12-step procedure
        print("Executing 12-step channel generation...")
        # Step 1: Set environment (done in __init__)
        
        # Step 2: Assign propagation condition
        # For now, let's assume a single UE for simplicity
        h_ut = self.config.ue_positions[0, 2]
        d_2d = np.linalg.norm(self.config.bs_positions[0, :2] - self.config.ue_positions[0, :2])
        is_los = self.condition_model.get_condition(d_2d, h_ut)
        
        # Step 3: Calculate pathloss
        pathloss_db = self.pathloss_model.calculate_pathloss(self.config, is_los)

        # Step 4: Generate large scale parameters
        fc_ghz = self.config.center_frequency_hz / 1e9
        lsps = self.fading_model.generate_lsps(is_los, fc_ghz)

        # Steps 5-11: Generate small-scale parameters and synthesize channel
        clusters = self._generate_clusters(lsps, is_los)
        H = self._synthesize_channel(clusters, lsps)

        # Step 12: Apply pathloss and shadowing
        total_loss_db = pathloss_db + lsps.shadow_fade_db
        H_final = H * 10**(-total_loss_db / 20)

        print(f"  - Condition: {'LOS' if is_los else 'NLOS'}")
        print(f"  - Pathloss: {pathloss_db:.2f} dB")
        print(f"  - LSPs: {lsps}")
        print("Channel generation complete.")

        return H_final

    def _generate_clusters(self, lsps: LargeScaleParameters, is_los: bool):
        """Generates small-scale parameters for all clusters (Steps 5-10)."""
        print("  - Generating small-scale parameters (clusters)...")
        scenario_params = self.params.get_scenario_params(self.config.scenario_name)
        num_clusters = scenario_params['num_clusters_los'] if is_los else scenario_params['num_clusters_nlos']
        
        # Step 5: Generate cluster delays
        r_tau = scenario_params['delay_scaling_factor']
        X_n = np.random.rand(num_clusters)
        tau_prime = -r_tau * lsps.delay_spread_s * np.log(X_n)
        tau_n = np.sort(tau_prime - np.min(tau_prime))

        # Step 6: Generate cluster powers
        P_prime = np.exp(-tau_n * (r_tau - 1) / (r_tau * lsps.delay_spread_s)) * 10**(-np.random.randn(num_clusters) / 10)
        P_n = P_prime / np.sum(P_prime)

        # Step 7: Generate arrival and departure angles
        # Azimuth angles (wrapped Gaussian)
        aoa_unscaled = np.sqrt(-np.log(P_n / np.max(P_n))) * (lsps.arrival_azimuth_spread_deg / 1.4)
        aod_unscaled = np.sqrt(-np.log(P_n / np.max(P_n))) * (lsps.departure_azimuth_spread_deg / 1.4)

        # Zenith angles (Laplacian)
        zoa_unscaled = -lsps.arrival_zenith_spread_deg * np.log(P_n / np.max(P_n))
        zod_unscaled = -lsps.departure_zenith_spread_deg * np.log(P_n / np.max(P_n))

        # Random signs and offsets
        X_n_signs = np.random.choice([-1, 1], size=(num_clusters, 4))
        Y_n_offsets = np.random.randn(num_clusters, 4)

        aoa_phi = X_n_signs[:, 0] * aoa_unscaled + Y_n_offsets[:, 0] * (lsps.arrival_azimuth_spread_deg / 7.0)
        aod_phi = X_n_signs[:, 1] * aod_unscaled + Y_n_offsets[:, 1] * (lsps.departure_azimuth_spread_deg / 7.0)
        zoa_theta = X_n_signs[:, 2] * zoa_unscaled + Y_n_offsets[:, 2] * (lsps.arrival_zenith_spread_deg / 7.0)
        zod_theta = X_n_signs[:, 3] * zod_unscaled + Y_n_offsets[:, 3] * (lsps.departure_zenith_spread_deg / 7.0)

        # TODO: Add LOS angle enforcement and offset angles from Table 7.5-3

        # Step 8: Couple rays within a cluster
        # For simplicity, we'll just use the generated angles directly for now.

        # Step 9: Generate XPR
        xpr_mu = scenario_params['xpr_mu_los'] if is_los else scenario_params['xpr_mu_nlos']
        xpr_sigma = scenario_params['xpr_sigma_los'] if is_los else scenario_params['xpr_sigma_nlos']
        X_n_xpr = np.random.randn(num_clusters)
        k_n = 10**((xpr_mu + X_n_xpr * xpr_sigma) / 10)

        # Step 10: Draw initial random phases
        initial_phases = np.random.uniform(-np.pi, np.pi, size=(num_clusters, 4))

        clusters = []
        for i in range(num_clusters):
            cluster = {
                'delay': tau_n[i],
                'power': P_n[i],
                'aod_phi': aod_phi[i],
                'aoa_phi': aoa_phi[i],
                'zod_theta': zod_theta[i],
                'zoa_theta': zoa_theta[i],
                'k_n': k_n[i],
                'initial_phases': initial_phases[i]
            }
            clusters.append(cluster)

        return clusters

    def _synthesize_channel(self, clusters, lsps):
        """Synthesizes the channel matrix H from cluster data (Step 11)."""
        print("  - Synthesizing channel coefficients...")
        num_rx = 1 # Placeholder
        num_tx = 1 # Placeholder
        H = np.zeros((num_rx, num_tx), dtype=np.complex128)

        c = 3e8 # Speed of light
        lambda_0 = c / self.config.center_frequency_hz
        
        num_rays_per_cluster = 20 # As per standard

        for cluster in clusters:
            # For each ray in the cluster
            for m in range(num_rays_per_cluster):
                # Simplified field patterns (assuming isotropic antennas)
                F_rx_u_theta = 1.0
                F_rx_u_phi = 1.0
                F_tx_s_theta = 1.0
                F_tx_s_phi = 1.0

                # Polarization matrix
                phi_theta_theta, phi_theta_phi, phi_phi_theta, phi_phi_phi = cluster['initial_phases']
                polarization_matrix = np.array([
                    [np.exp(1j * phi_theta_theta), np.sqrt(1/cluster['k_n']) * np.exp(1j * phi_theta_phi)],
                    [np.sqrt(1/cluster['k_n']) * np.exp(1j * phi_phi_theta), np.exp(1j * phi_phi_phi)]
                ])

                # Spherical unit vectors (simplified)
                r_rx = np.array([1, 0, 0])
                r_tx = np.array([1, 0, 0])

                # Doppler component (simplified, assuming static)
                v = self.config.ue_velocities[0]
                doppler_term = np.exp(1j * 2 * np.pi * (np.dot(r_rx, v)) / lambda_0 * 0) # t=0

                # Antenna element location vectors (simplified)
                d_rx_u = np.array([0, 0, 0])
                d_tx_s = np.array([0, 0, 0])

                # Spatial signatures
                spatial_sig_rx = np.array([F_rx_u_theta, F_rx_u_phi])
                spatial_sig_tx = np.array([F_tx_s_theta, F_tx_s_phi])

                # Phase term due to spatial separation
                phase_term = np.exp(1j * 2 * np.pi * (np.dot(r_rx, d_rx_u) + np.dot(r_tx, d_tx_s)) / lambda_0)

                # Combine everything for this ray
                ray_contribution = np.sqrt(cluster['power'] / num_rays_per_cluster) * \
                                   np.dot(spatial_sig_rx, np.dot(polarization_matrix, spatial_sig_tx.T)) * \
                                   doppler_term * phase_term

                H[0, 0] += ray_contribution

        return H