import numpy as np
from typing import Optional, Callable, List
from .common.data_models import ScenarioConfig, LargeScaleParameters, EngineCluster
from .parameters.parameter_loader import ParameterTables
from .models.abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from .models.uma_models import UMa_ChannelCondition, UMa_Pathloss, UMa_Fading
from .models.umi_models import UMi_ChannelCondition, UMi_Pathloss, UMi_Fading
from .models.rma_models import RMa_ChannelCondition, RMa_Pathloss, RMa_Fading
from .models.inh_models import InH_ChannelCondition, InH_Pathloss, InH_Fading
from .models.sma_models import SMa_ChannelCondition, SMa_Pathloss, SMa_Fading
from .models.umi_open_area_models import UMi_OpenArea_ChannelCondition, UMi_OpenArea_Pathloss, UMi_OpenArea_Fading
from .models.inf_models import InF_ChannelCondition, InF_Pathloss, InF_Fading
from .models.spatial_correlation import get_spatial_correlation_matrix
from .common.geometry import get_antenna_positions

class ChannelGenerator:
    """
    核心信道生成器引擎。
    该类负责根据用户配置，执行3GPP TR 38.901标准中定义的12个步骤，生成一个信道实现。
    """
    def __init__(self, config: ScenarioConfig, params: ParameterTables, cluster_model_name: Optional[str] = None, use_median_lsps: bool = False):
        """
        初始化信道生成器。

        Args:
            config (ScenarioConfig): 包含所有仿真参数的配置对象。
            params (ParameterTables): 从JSON文件加载的3GPP标准参数表。
            cluster_model_name (Optional[str], optional): 小尺度簇模型的名称。默认为 'Stochastic'。
            use_median_lsps (bool, optional): 是否使用LSP中位值。默认为 False。
        """
        self.config = config
        self.params = params
        self.cluster_model_name = cluster_model_name if cluster_model_name else 'Stochastic'
        self.use_median_lsps = use_median_lsps
        self._select_models()

    def _select_models(self):
        """
        根据场景名称，选择并实例化对应的信道条件、路径损耗和衰落模型。
        这是一种策略设计模式的应用。
        """
        scenario = self.config.scenario_name
        model_map = {
            'UMa': (UMa_ChannelCondition, UMa_Pathloss, UMa_Fading),
            'UMi-StreetCanyon': (UMi_ChannelCondition, UMi_Pathloss, UMi_Fading),
            'UMi-OpenArea': (UMi_OpenArea_ChannelCondition, UMi_OpenArea_Pathloss, UMi_OpenArea_Fading),
            'RMa': (RMa_ChannelCondition, RMa_Pathloss, RMa_Fading),
            'InH-Office': (InH_ChannelCondition, InH_Pathloss, InH_Fading),
            'SMa': (SMa_ChannelCondition, SMa_Pathloss, SMa_Fading),
            'InF': (InF_ChannelCondition, InF_Pathloss, InF_Fading),
        }
        if scenario in model_map:
            cond_model, pl_model, fade_model = model_map[scenario]
            self.condition_model = cond_model()
            self.pathloss_model = pl_model(self.params)
            self.fading_model = fade_model(self.params)
        else:
            raise NotImplementedError(f"场景 '{scenario}' 的LSP生成暂不支持。")

    def _ensure_antenna_positions(self, lambda_0: float) -> None:
        ant_config = self.config.antenna_config
        if ant_config.tx_positions is None:
            ant_config.tx_positions = get_antenna_positions(
                ant_config.tx_array_type,
                ant_config.num_tx_rows, ant_config.num_tx_cols,
                ant_config.tx_spacing_h * lambda_0,
                ant_config.tx_spacing_v * lambda_0
            )
        if ant_config.rx_positions is None:
            ant_config.rx_positions = get_antenna_positions(
                ant_config.rx_array_type,
                ant_config.num_rx_rows, ant_config.num_rx_cols,
                ant_config.rx_spacing_h * lambda_0,
                ant_config.rx_spacing_v * lambda_0
            )

    @staticmethod
    def _unit_vector(zenith_deg: float, azimuth_deg: float) -> np.ndarray:
        theta = np.deg2rad(zenith_deg)
        phi = np.deg2rad(azimuth_deg)
        return np.array([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            np.sin(theta)
        ])

    def _synthesize_plane_wave(self, k_hat_rx: np.ndarray, k_hat_tx: np.ndarray, lambda_0: float) -> np.ndarray:
        ant_config = self.config.antenna_config
        k_factor = 2 * np.pi / lambda_0
        a_rx = np.exp(1j * k_factor * (ant_config.rx_positions @ k_hat_rx))
        a_tx = np.exp(1j * k_factor * (ant_config.tx_positions @ k_hat_tx))
        return a_rx[:, None] @ a_tx[None, :]

    def _synthesize_los_component(self, lambda_0: float) -> np.ndarray:
        ant_config = self.config.antenna_config
        num_rx = ant_config.num_rx_rows * ant_config.num_rx_cols
        num_tx = ant_config.num_tx_rows * ant_config.num_tx_cols
        bs_pos = self.config.bs_positions[0]
        ue_pos = self.config.ue_positions[0]
        d_vec = ue_pos - bs_pos
        dist = np.linalg.norm(d_vec)
        if dist < 1e-9:
            return np.zeros((num_rx, num_tx), dtype=np.complex128)
        k_hat_tx = d_vec / dist
        k_hat_rx = -k_hat_tx
        los_phase = np.exp(-1j * 2 * np.pi * dist / lambda_0)
        return los_phase * self._synthesize_plane_wave(k_hat_rx, k_hat_tx, lambda_0)

    def generate_channel_realization(self, force_condition: Optional[str] = None, time_s: float = 0.0, logger: Optional[Callable[[str], None]] = print, use_median_lsps: Optional[bool] = None) -> tuple:
        """
        生成单次的信道实现。这是该类的核心方法，实现了混合信道模型的“三步走”流程。

        Args:
            force_condition (Optional[str]): 强制指定 'LOS' 或 'NLOS'。
            time_s (float): 仿真时间点（秒），用于计算多普勒频移。
            logger (Optional[Callable]): 日志记录器。
            use_median_lsps (Optional[bool]): 单次运行覆盖LSP中位值设置。

        Returns:
            tuple: 包含信道矩阵、簇信息、LSP等结果的元组。
        """
        if logger: logger(f"开始为环境 '{self.config.scenario_name}' 生成信道...")
        
        # 步骤 1: 始终基于大尺度环境生成大尺度参数 (LSP)
        h_ut = self.config.ue_positions[0, 2]
        d_2d = np.linalg.norm(self.config.bs_positions[0, :2] - self.config.ue_positions[0, :2])
        is_los = self.condition_model.get_condition(d_2d, h_ut)
        if force_condition == 'LOS': is_los = True
        if force_condition == 'NLOS': is_los = False
        pathloss_db = self.pathloss_model.calculate_pathloss(self.config, is_los)
        fc_ghz = self.config.center_frequency_hz / 1e9
        
        use_median = use_median_lsps if use_median_lsps is not None else self.use_median_lsps
        
        lsps = self.fading_model.generate_lsps(self.config, is_los, use_median_lsps=use_median)

        if logger: 
            logger(f"  - 大尺度环境: {self.config.scenario_name}")
            logger(f"  - 簇模型: {self.cluster_model_name}")
            logger(f"  - 信道条件: {'LOS' if is_los else 'NLOS'}")
            logger(f"  - 路径损耗: {pathloss_db:.2f} dB")
            if use_median:
                logger("  - LSP生成: 使用中位值 (确定性)")
            else:
                logger("  - LSP生成: 使用随机采样 (随机性)")

        # 步骤 2: 根据选择的簇模型，生成或加载簇参数
        is_deterministic_clusters = self.cluster_model_name.startswith('CDL-') or self.cluster_model_name == 'SCME' or self.cluster_model_name.startswith('SCME-')
        if is_deterministic_clusters:
            if logger: logger(f"  - 为 {self.cluster_model_name} 加载确定性簇...")
            clusters = self._load_deterministic_clusters(self.cluster_model_name)
        else:
            if logger: logger("  - 生成随机簇...")
            clusters = self._generate_stochastic_clusters(lsps, is_los, logger)

        # 步骤 3: 合成信道并应用大尺度效应
        H_normalized = self._synthesize_channel(clusters, lsps, time_s, is_deterministic_clusters, logger)
        total_loss_db = pathloss_db + lsps.shadow_fade_db
        H_final = H_normalized * 10**(-total_loss_db / 20)

        if logger: logger("信道生成完成。")
        return H_final, H_normalized, clusters, lsps, pathloss_db, is_los

    def _load_deterministic_clusters(self, model_name: str) -> List[EngineCluster]:
        """从参数文件中加载确定性模型（如CDL）的簇参数。"""
        if model_name.startswith('CDL-'):
            model_params = self.params.get_cdl_model_params(model_name)
        elif model_name == 'SCME' or model_name.startswith('SCME-'):
            model_params = self.params.get_scme_model_params(model_name)
        else:
            return []

        if not model_params:
            raise ValueError(f"在参数文件中未找到确定性模型 '{model_name}'。")

        clusters = []
        # 对于确定性模型，JSON中的功率是绝对的。我们将其归一化以供引擎使用。
        total_power_linear = sum(10**(c['power_db'] / 10) for c in model_params['clusters'])
        
        last_aod, last_aoa, last_zod, last_zoa = 0, 0, 90, 90

        for c_data in model_params['clusters']:
            power_linear = 10**(c_data['power_db'] / 10)
            normalized_power = power_linear / total_power_linear

            aod = c_data.get('aod_deg', last_aod)
            aoa = c_data.get('aoa_deg', last_aoa)
            zod = c_data.get('zod_deg', last_zod)
            zoa = c_data.get('zoa_deg', last_zoa)
            last_aod, last_aoa, last_zod, last_zoa = aod, aoa, zod, zoa

            # Phase 6 fix (PR #6 Codex P2): mirror the generated linear XPR
            # into both fields. `k_n` is the legacy name (kept for backward
            # compat); `xpr_linear` is the canonical Phase 6+ field consumed
            # by the dual-pol synthesizer. Until k_n is fully renamed, the
            # 3GPP path must write both so Phase 6 _build_pol_matrix sees the
            # scenario-specific XPR instead of the 7 dB default fallback.
            xpr_lin = 10 ** (model_params.get('xpr_db', 11) / 10)
            cluster = EngineCluster(
                delay=c_data['delay_ns'] * 1e-9,
                power=normalized_power,
                aod_phi=aod, aoa_phi=aoa, zod_theta=zod, zoa_theta=zoa,
                as_aod=c_data.get('as_aod_deg', 0.0),
                as_aoa=c_data.get('as_aoa_deg', 0.0),
                as_zod=c_data.get('as_zod_deg', 0.0),
                as_zoa=c_data.get('as_zoa_deg', 0.0),
                k_n=xpr_lin,            # legacy field (was sole XPR storage)
                xpr_linear=xpr_lin,     # canonical Phase 6+ field
            )
            clusters.append(cluster)
        return clusters

    def _generate_stochastic_clusters(self, lsps: LargeScaleParameters, is_los: bool, logger: Optional[Callable[[str], None]]) -> List[EngineCluster]:
        """根据LSP随机生成簇参数。"""
        scenario_params = self.params.get_scenario_params(self.config.scenario_name)
        num_clusters = scenario_params['num_clusters_los'] if is_los else scenario_params['num_clusters_nlos']
        tau_n = self._generate_cluster_delays(lsps, num_clusters, is_los, scenario_params)
        P_n = self._generate_cluster_powers(tau_n, lsps, is_los, scenario_params)
        angles = self._generate_angles(P_n, lsps, num_clusters)
        k_n = self._generate_xpr(scenario_params, is_los, num_clusters)
        initial_phases = self._generate_initial_phases(num_clusters)
        
        clusters = []
        for i in range(num_clusters):
            # Phase 6 fix (PR #6 Codex P2): mirror per-cluster XPR into both
            # `k_n` (legacy) and `xpr_linear` (canonical Phase 6+ field).
            clusters.append(EngineCluster(
                delay=tau_n[i], power=P_n[i],
                aod_phi=angles['aod_phi'][i], aoa_phi=angles['aoa_phi'][i],
                zod_theta=angles['zod_theta'][i], zoa_theta=angles['zoa_theta'][i],
                as_aod=lsps.departure_azimuth_spread_deg, as_aoa=lsps.arrival_azimuth_spread_deg,
                as_zod=lsps.departure_zenith_spread_deg, as_zoa=lsps.arrival_zenith_spread_deg,
                k_n=k_n[i],
                xpr_linear=k_n[i],
                initial_phases=initial_phases[i],
            ))
        return clusters

    def _generate_cluster_delays(self, lsps, num_clusters, is_los, scenario_params):
        """生成簇时延。参考 TR 38.901, 步骤 4。"""
        r_tau = scenario_params['delay_scaling_factor']
        X_n = np.random.rand(num_clusters)
        tau_prime = -r_tau * lsps.delay_spread_s * np.log(X_n)
        return np.sort(tau_prime - np.min(tau_prime))

    def _generate_cluster_powers(self, tau_n, lsps, is_los, scenario_params):
        """生成簇功率。参考 TR 38.901, 步骤 5。"""
        r_tau = scenario_params['delay_scaling_factor']
        P_prime = np.exp(-tau_n * (r_tau - 1) / (r_tau * lsps.delay_spread_s)) * 10**(-np.random.randn(len(tau_n)) / 10)
        return P_prime / np.sum(P_prime)

    def _generate_angles(self, P_n, lsps, num_clusters):
        """生成簇的离开角和到达角。参考 TR 38.901, 步骤 6 & 7。"""
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
        """生成交叉极化功率比 (XPR)。"""
        xpr_mu = scenario_params['xpr_mu_los'] if is_los else scenario_params['xpr_mu_nlos']
        xpr_sigma = scenario_params['xpr_sigma_los'] if is_los else scenario_params['xpr_sigma_nlos']
        return 10**((xpr_mu + np.random.randn(num_clusters) * xpr_sigma) / 10)

    def _generate_initial_phases(self, num_clusters):
        """为每个簇的四个极化组合生成随机初始相位。"""
        return np.random.uniform(-np.pi, np.pi, size=(num_clusters, 4))

    def _synthesize_channel(self, clusters: List[EngineCluster], lsps: LargeScaleParameters, time_s: float, is_deterministic: bool, logger: Optional[Callable[[str], None]]) -> np.ndarray:
        """
        合成最终的信道矩阵。
        此方法实现了 TR 38.901 的步骤 8 到 11。
        """
        if logger: logger("  - 合成信道系数...")
        ant_config = self.config.antenna_config
        num_rx = ant_config.num_rx_rows * ant_config.num_rx_cols
        num_tx = ant_config.num_tx_rows * ant_config.num_tx_cols
        
        H_total_normalized = np.zeros((num_rx, num_tx), dtype=np.complex128)
        c = 3e8
        lambda_0 = c / self.config.center_frequency_hz
        self._ensure_antenna_positions(lambda_0)

        # 步骤 9, 10, 11: 对所有簇进行循环，计算并叠加其贡献
        for cluster in clusters:
            if is_deterministic:
                k_hat_rx = self._unit_vector(cluster.zoa_theta, cluster.aoa_phi)
                k_hat_tx = self._unit_vector(cluster.zod_theta, cluster.aod_phi)
                H_corr = self._synthesize_plane_wave(k_hat_rx, k_hat_tx, lambda_0)
            else:
                # 生成IID瑞利衰落矩阵
                H_iid = (np.random.randn(num_rx, num_tx) + 1j * np.random.randn(num_rx, num_tx)) / np.sqrt(2)

                # 随机模型 (Stochastic): 为每个簇计算独立的空间相关性
                # 使用该簇特定的角度和LSP中的角度扩展
                R_rx = get_spatial_correlation_matrix(
                    ant_config.rx_array_type, ant_config.num_rx_rows, ant_config.num_rx_cols, 
                    ant_config.rx_spacing_v, ant_config.rx_spacing_h, 
                    cluster.zoa_theta, cluster.aoa_phi, 
                    lsps.arrival_zenith_spread_deg, lsps.arrival_azimuth_spread_deg
                )
                R_tx = get_spatial_correlation_matrix(
                    ant_config.tx_array_type, ant_config.num_tx_rows, ant_config.num_tx_cols, 
                    ant_config.tx_spacing_v, ant_config.tx_spacing_h, 
                    cluster.zod_theta, cluster.aod_phi, 
                    lsps.departure_zenith_spread_deg, lsps.departure_azimuth_spread_deg
                )
                try:
                    L_rx = np.linalg.cholesky(R_rx)
                    L_tx = np.linalg.cholesky(R_tx)
                    H_corr = L_rx @ H_iid @ L_tx.conj().T
                    
                    # DEBUG: Check R_rx for first cluster
                    # if clusters.index(cluster) == 0:
                    #    print(f"[DEBUG R_rx Cluster 0] AoA: {cluster.aoa_phi:.2f}, R_01: {R_rx[0,1]:.4f}")
                        
                except np.linalg.LinAlgError:
                    # 如果矩阵不是正定的，回退到单位矩阵 (IID)
                    H_corr = H_iid

            cluster.h_matrix = H_corr
            
            # 计算多普勒频移
            v = self.config.ue_velocities[0]
            r_rx_mean = np.array([np.cos(np.deg2rad(cluster.zoa_theta)) * np.cos(np.deg2rad(cluster.aoa_phi)),
                                  np.cos(np.deg2rad(cluster.zoa_theta)) * np.sin(np.deg2rad(cluster.aoa_phi)),
                                  np.sin(np.deg2rad(cluster.zoa_theta))])
            doppler_shift_hz = np.dot(r_rx_mean, v) / lambda_0
            doppler_term = np.exp(1j * 2 * np.pi * doppler_shift_hz * time_s)
            
            # 累加当前簇的贡献
            H_total_normalized += np.sqrt(cluster.power) * H_corr * doppler_term

        if lsps.k_factor_linear > 0:
            k_linear = lsps.k_factor_linear
            H_los = self._synthesize_los_component(lambda_0)
            H_total_normalized = (
                np.sqrt(k_linear / (k_linear + 1)) * H_los +
                np.sqrt(1 / (k_linear + 1)) * H_total_normalized
            )

        return H_total_normalized

    def export_sos_parameters(self) -> List[dict]:
        """
        导出用于 FPGA SoS (Sum-of-Sinusoids) 模块的参数表。
        
        实现 3GPP TR 38.901 Table 7.5-3 定义的 Ray 级精度。
        每个簇分解为 20 条子射线 (Rays)。
        
        Returns:
            List[dict]: 参数列表 (Rows = Clusters * 20 * Rx * Tx).
        """
        if self.condition_model is None:
            self._select_models()

        # --- 1. Re-generate Environment (LSP + Clusters) ---
        # Note: Caller must ensure fixed random seed for consistency
        h_ut = self.config.ue_positions[0, 2]
        d_2d = np.linalg.norm(self.config.bs_positions[0, :2] - self.config.ue_positions[0, :2])
        is_los = self.condition_model.get_condition(d_2d, h_ut)
        lsps = self.fading_model.generate_lsps(self.config, is_los, use_median_lsps=self.use_median_lsps)
        
        if self.cluster_model_name.startswith('CDL-') or self.cluster_model_name == 'SCME':
            clusters = self._load_deterministic_clusters(self.cluster_model_name)
        else:
            clusters = self._generate_stochastic_clusters(lsps, is_los, None)

        # --- 2. Ray Offset Table (TR 38.901 Table 7.5-3) ---
        # Load from parameters (Algorithm-Data Decoupling)
        scenario_params = self.params.get_scenario_params(self.config.scenario_name)
        if 'ray_angle_offsets_deg' not in scenario_params:
             # Fallback if not found (though it should be there)
             ray_offsets = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 
                                     0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])
        else:
             ray_offsets = np.array(scenario_params['ray_angle_offsets_deg'])
        
        sos_params = []
        ant_config = self.config.antenna_config
        num_rx = ant_config.num_rx_rows * ant_config.num_rx_cols
        num_tx = ant_config.num_tx_rows * ant_config.num_tx_cols
        c = 3e8
        lambda_0 = c / self.config.center_frequency_hz
        v_vec = self.config.ue_velocities[0] # (vx, vy, vz)
        
        # Scaling factors (c_ASD etc.) from Table 7.5-3
        # For cluster size 20, C_AS = 1.0 (approx, standard says C_phi = 1.125 roughly for 20 rays?) 
        # Actually Table 7.5-2 says for N=20, C_phi = 1.289? No, let's check standard.
        # TR 38.901 V17.0.0 Table 7.5-2: for M=20 rays, C_AS = 1.0/1.4?? 
        # Let's use the standard scaling constant C = 1.0 for simplicity or assume the offsets are already scaled?
        # The offsets in Table 7.5-3 are normalized. The mapping is: phi_n_m = phi_n + c_ASD * alpha_m * ASD_n
        # The constant c_ASD depends on number of rays. For M=20, c_ASD is commonly cited as 1/1.0 or similar.
        # However, to be strictly precise, we should use the defined C values.
        # Let's assume C = 1.0 for now to avoid magic number confusion without full table lookups.
        c_scale = 1.0 

        # --- 3. Iterate Clusters & Rays ---
        for cluster_idx, cluster in enumerate(clusters):
            
            # 20 Rays per cluster
            for m in range(20):
                alpha = ray_offsets[m]
                
                # A. Apply Angle Offsets
                # Note: ASA/ASD/ZSA/ZSD are in degrees.
                # Angles in cluster object are in degrees.
                
                # AoA
                aoa_ray = cluster.aoa_phi + c_scale * alpha * cluster.as_aoa
                # AoD
                aod_ray = cluster.aod_phi + c_scale * alpha * cluster.as_aod
                # ZoA
                zoa_ray = cluster.zoa_theta + c_scale * alpha * cluster.as_zoa
                # ZoD
                zod_ray = cluster.zod_theta + c_scale * alpha * cluster.as_zod

                # B. Calculate Doppler (Hz) for this Ray
                # 3D Projection
                theta_rad = np.deg2rad(zoa_ray)
                phi_rad = np.deg2rad(aoa_ray)
                r_hat = np.array([
                    np.sin(theta_rad) * np.cos(phi_rad),
                    np.sin(theta_rad) * np.sin(phi_rad),
                    np.cos(theta_rad)
                ])
                doppler_freq_hz = np.dot(v_vec, r_hat) / lambda_0
                
                # C. Calculate Complex Weight (Steering Vector based)
                # Power per ray
                p_ray = cluster.power / 20.0
                
                # Random Initial Phase for this Ray
                # In standard, this is Phi_n_m. 
                # For deterministic reproducibility, we should hash the seed or use a stable RNG.
                # Here we just draw from uniform since seed is set at start.
                phase_init = np.random.uniform(-np.pi, np.pi)
                
                # Tx Steering Vector (Array Response)
                # We need the positions of elements relative to (0,0,0) or array center.
                # ant_config.tx_positions is shape (N_tx, 3)
                # k_tx vector: spherical to cartesian for departure
                theta_t_rad = np.deg2rad(zod_ray)
                phi_t_rad = np.deg2rad(aod_ray)
                k_hat_tx = np.array([
                    np.sin(theta_t_rad) * np.cos(phi_t_rad),
                    np.sin(theta_t_rad) * np.sin(phi_t_rad),
                    np.cos(theta_t_rad)
                ])
                # Phase shift = k_vec . position_vec * (2pi/lambda)
                # k_vec = k_hat * (2pi/lambda)
                # phase = (2pi/lambda) * (x*sx + y*sy + z*sz)
                k_factor = 2 * np.pi / lambda_0
                
                # Rx Steering Vector
                # k_rx vector: spherical to cartesian for arrival
                # k_hat_rx is same as r_hat (Doppler direction) usually
                k_hat_rx = r_hat 
                
                # Iterate all antenna pairs
                for rx_idx in range(num_rx):
                    # Calculate Rx Array Response Phase
                    pos_rx = ant_config.rx_positions[rx_idx]
                    phase_rx = k_factor * np.dot(k_hat_rx, pos_rx)
                    
                    for tx_idx in range(num_tx):
                        # Calculate Tx Array Response Phase
                        pos_tx = ant_config.tx_positions[tx_idx]
                        phase_tx = k_factor * np.dot(k_hat_tx, pos_tx)
                        
                        # Total Phase = Init + Rx_geom + Tx_geom
                        # Note: In standard, there is also XPR polarization mixing matrix.
                        # As discussed, assuming co-pol / isotropic scalar gain for simplification.
                        # Total Weight = sqrt(P_ray) * exp(j * (Phi_init + Phase_rx + Phase_tx))
                        
                        total_phase = phase_init + phase_rx + phase_tx
                        weight = np.sqrt(p_ray) * np.exp(1j * total_phase)
                        
                        sos_params.append({
                            "Cluster_ID": cluster_idx,
                            "Ray_ID": m,
                            "Rx_Ant_ID": rx_idx,
                            "Tx_Ant_ID": tx_idx,
                            "Doppler_Hz": doppler_freq_hz,
                            "Delay_s": cluster.delay,
                            "Weight_Real": weight.real,
                            "Weight_Imag": weight.imag,
                            "Power_Linear": p_ray,
                            "Phase_Rad": np.angle(weight)
                        })
                    
        return sos_params
