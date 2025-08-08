import numpy as np
from typing import Optional
from .common.data_models import ScenarioConfig, LargeScaleParameters
from .parameters.parameter_loader import ParameterTables
from .models.abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from .models.uma_models import UMa_ChannelCondition, UMa_Pathloss, UMa_Fading
from .models.umi_models import UMi_ChannelCondition, UMi_Pathloss, UMi_Fading
from .models.rma_models import RMa_ChannelCondition, RMa_Pathloss, RMa_Fading
from .models.inh_models import InH_ChannelCondition, InH_Pathloss, InH_Fading
from .models.sma_models import SMa_ChannelCondition, SMa_Pathloss, SMa_Fading
from .models.umi_open_area_models import UMi_OpenArea_ChannelCondition, UMi_OpenArea_Pathloss, UMi_OpenArea_Fading
from .models.inf_models import InF_ChannelCondition, InF_Pathloss, InF_Fading

class ChannelGenerator:
    """
    Core simulation engine that generates channel realizations based on TR 38.901.
    """
    def __init__(self, config: ScenarioConfig, params: ParameterTables):
        """Initializes the ChannelGenerator.

        Args:
            config: The simulation configuration.
            params: The parameter tables.
        """
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
        elif scenario == 'UMi-OpenArea':
            self.condition_model = UMi_OpenArea_ChannelCondition()
            self.pathloss_model = UMi_OpenArea_Pathloss(self.params)
            self.fading_model = UMi_OpenArea_Fading(self.params)
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
        elif scenario == 'InF':
            self.condition_model = InF_ChannelCondition()
            self.pathloss_model = InF_Pathloss(self.params)
            self.fading_model = InF_Fading(self.params)
        else:
            raise NotImplementedError(f"Scenario '{scenario}' is not yet supported.")

    def generate_channel_realization(self, force_condition: Optional[str] = None):
        """Executes the 12-step channel generation procedure from TR 38.901 Section 7.5."""
        print("Executing 12-step channel generation...")
        
        if force_condition == 'LOS':
            is_los = True
            print("  - Forcing LOS condition.")
        elif force_condition == 'NLOS':
            is_los = False
            print("  - Forcing NLOS condition.")
        else:
            h_ut = self.config.ue_positions[0, 2]
            d_2d = np.linalg.norm(self.config.bs_positions[0, :2] - self.config.ue_positions[0, :2])
            is_los = self.condition_model.get_condition(d_2d, h_ut)
        
        pathloss_db = self.pathloss_model.calculate_pathloss(self.config, is_los)

        fc_ghz = self.config.center_frequency_hz / 1e9
        lsps = self.fading_model.generate_lsps(is_los, fc_ghz)

        clusters = self._generate_clusters(lsps, is_los)
        H = self._synthesize_channel(clusters, lsps)

        total_loss_db = pathloss_db + lsps.shadow_fade_db
        H_final = H * 10**(-total_loss_db / 20)

        print(f"  - Condition: {'LOS' if is_los else 'NLOS'}")
        print(f"  - Pathloss: {pathloss_db:.2f} dB")
        print(f"  - LSPs: {lsps}")
        print("Channel generation complete.")

        return H_final, clusters, lsps, pathloss_db, is_los

    def _generate_clusters(self, lsps: LargeScaleParameters, is_los: bool):
        """Generates small-scale parameters for all clusters (Steps 5-10)."""
        print("  - Generating small-scale parameters (clusters)...")
        scenario_params = self.params.get_scenario_params(self.config.scenario_name)
        num_clusters = scenario_params['num_clusters_los'] if is_los else scenario_params['num_clusters_nlos']
        
        tau_n = self._generate_cluster_delays(lsps, num_clusters, is_los, scenario_params)
        P_n = self._generate_cluster_powers(tau_n, lsps, is_los, scenario_params)
        angles = self._generate_angles(P_n, lsps, num_clusters)
        k_n = self._generate_xpr(scenario_params, is_los, num_clusters)
        initial_phases = self._generate_initial_phases(num_clusters)

        clusters = []
        for i in range(num_clusters):
            cluster = {
                'delay': tau_n[i],
                'power': P_n[i],
                'aod_phi': angles['aod_phi'][i],
                'aoa_phi': angles['aoa_phi'][i],
                'zod_theta': angles['zod_theta'][i],
                'zoa_theta': angles['zoa_theta'][i],
                'k_n': k_n[i],
                'initial_phases': initial_phases[i]
            }
            clusters.append(cluster)

        return clusters

    def _generate_cluster_delays(self, lsps, num_clusters, is_los, scenario_params):
        """Generates cluster delays (Step 5)."""
        r_tau = scenario_params['delay_scaling_factor']
        X_n = np.random.rand(num_clusters)
        tau_prime = -r_tau * lsps.delay_spread_s * np.log(X_n)
        return np.sort(tau_prime - np.min(tau_prime))

    def _generate_cluster_powers(self, tau_n, lsps, is_los, scenario_params):
        """Generates cluster powers (Step 6)."""
        r_tau = scenario_params['delay_scaling_factor']
        P_prime = np.exp(-tau_n * (r_tau - 1) / (r_tau * lsps.delay_spread_s)) * 10**(-np.random.randn(len(tau_n)) / 10)
        return P_prime / np.sum(P_prime)

    def _generate_angles(self, P_n, lsps, num_clusters):
        """Generates arrival and departure angles (Step 7)."""
        aoa_unscaled = np.sqrt(-np.log(P_n / np.max(P_n))) * (lsps.arrival_azimuth_spread_deg / 1.4)
        aod_unscaled = np.sqrt(-np.log(P_n / np.max(P_n))) * (lsps.departure_azimuth_spread_deg / 1.4)
        zoa_unscaled = -lsps.arrival_zenith_spread_deg * np.log(P_n / np.max(P_n))
        zod_unscaled = -lsps.departure_zenith_spread_deg * np.log(P_n / np.max(P_n))

        X_n_signs = np.random.choice([-1, 1], size=(num_clusters, 4))
        Y_n_offsets = np.random.randn(num_clusters, 4)

        return {
            'aoa_phi': X_n_signs[:, 0] * aoa_unscaled + Y_n_offsets[:, 0] * (lsps.arrival_azimuth_spread_deg / 7.0),
            'aod_phi': X_n_signs[:, 1] * aod_unscaled + Y_n_offsets[:, 1] * (lsps.departure_azimuth_spread_deg / 7.0),
            'zoa_theta': X_n_signs[:, 2] * zoa_unscaled + Y_n_offsets[:, 2] * (lsps.arrival_zenith_spread_deg / 7.0),
            'zod_theta': X_n_signs[:, 3] * zod_unscaled + Y_n_offsets[:, 3] * (lsps.departure_zenith_spread_deg / 7.0)
        }

    def _generate_xpr(self, scenario_params, is_los, num_clusters):
        """Generates cross-polarization power ratios (Step 9)."""
        xpr_mu = scenario_params['xpr_mu_los'] if is_los else scenario_params['xpr_mu_nlos']
        xpr_sigma = scenario_params['xpr_sigma_los'] if is_los else scenario_params['xpr_sigma_nlos']
        return 10**((xpr_mu + np.random.randn(num_clusters) * xpr_sigma) / 10)

    def _generate_initial_phases(self, num_clusters):
        """Draws initial random phases (Step 10)."""
        return np.random.uniform(-np.pi, np.pi, size=(num_clusters, 4))

    def _synthesize_channel(self, clusters, lsps):
        """Synthesizes the channel matrix H from cluster data (Step 11)."""
        print("  - Synthesizing channel coefficients...")
        num_rx = self.config.antenna_config.num_rx_antennas
        num_tx = self.config.antenna_config.num_tx_antennas
        H = np.zeros((num_rx, num_tx), dtype=np.complex128)

        c = 3e8 # Speed of light
        lambda_0 = c / self.config.center_frequency_hz
        
        num_rays_per_cluster = 20 # As per standard

        for u in range(num_rx):
            for s in range(num_tx):
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

                        # Antenna element location vectors
                        d_rx_u = self.config.antenna_config.rx_positions[u]
                        d_tx_s = self.config.antenna_config.tx_positions[s]

                        # Spatial signatures
                        spatial_sig_rx = np.array([F_rx_u_theta, F_rx_u_phi])
                        spatial_sig_tx = np.array([F_tx_s_theta, F_tx_s_phi])

                        # Phase term due to spatial separation
                        phase_term = np.exp(1j * 2 * np.pi * (np.dot(r_rx, d_rx_u) + np.dot(r_tx, d_tx_s)) / lambda_0)

                        # Combine everything for this ray
                        ray_contribution = np.sqrt(cluster['power'] / num_rays_per_cluster) * \
                                           np.dot(spatial_sig_rx, np.dot(polarization_matrix, spatial_sig_tx.T)) * \
                                           doppler_term * phase_term

                        H[u, s] += ray_contribution

        # Normalize the channel matrix to have unit power
        H /= np.sqrt(np.mean(np.abs(H)**2))

        return H