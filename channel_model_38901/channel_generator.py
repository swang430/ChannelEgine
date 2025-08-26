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
        
        lsps = self.fading_model.generate_lsps(is_los, fc_ghz, use_median_lsps=use_median)

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

            cluster = EngineCluster(
                delay=c_data['delay_ns'] * 1e-9,
                power=normalized_power,
                aod_phi=aod, aoa_phi=aoa, zod_theta=zod, zoa_theta=zoa,
                as_aod=c_data.get('as_aod_deg', 0.0),
                as_aoa=c_data.get('as_aoa_deg', 0.0),
                as_zod=c_data.get('as_zod_deg', 0.0),
                as_zoa=c_data.get('as_zoa_deg', 0.0),
                k_n=10**(model_params.get('xpr_db', 11) / 10) # 使用模型全局的XPR
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
            clusters.append(EngineCluster(
                delay=tau_n[i], power=P_n[i],
                aod_phi=angles['aod_phi'][i], aoa_phi=angles['aoa_phi'][i],
                zod_theta=angles['zod_theta'][i], zoa_theta=angles['zoa_theta'][i],
                as_aod=lsps.departure_azimuth_spread_deg, as_aoa=lsps.arrival_azimuth_spread_deg,
                as_zod=lsps.departure_zenith_spread_deg, as_zoa=lsps.arrival_zenith_spread_deg,
                k_n=k_n[i], initial_phases=initial_phases[i]
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
        
        # 步骤 8: 生成空间相关矩阵
        if is_deterministic:
            # 确定性模型（如CDL）不使用空间相关性
            L_rx, L_tx = np.eye(num_rx), np.eye(num_tx)
        else:
            mean_aoa = clusters[0].aoa_phi; mean_zoa = clusters[0].zoa_theta
            mean_aod = clusters[0].aod_phi; mean_zod = clusters[0].zod_theta
            R_rx = get_spatial_correlation_matrix(ant_config.rx_array_type, ant_config.num_rx_rows, ant_config.num_rx_cols, ant_config.rx_spacing_v, ant_config.rx_spacing_h, mean_zoa, mean_aoa, lsps.arrival_azimuth_spread_deg, lsps.arrival_azimuth_spread_deg)
            R_tx = get_spatial_correlation_matrix(ant_config.tx_array_type, ant_config.num_tx_rows, ant_config.num_tx_cols, ant_config.tx_spacing_v, ant_config.tx_spacing_h, mean_zod, mean_aod, lsps.departure_zenith_spread_deg, lsps.departure_azimuth_spread_deg)
            try:
                L_rx = np.linalg.cholesky(R_rx)
                L_tx = np.linalg.cholesky(R_tx)
            except np.linalg.LinAlgError:
                # 如果矩阵不是正定的，则使用单位矩阵
                L_rx = np.eye(num_rx)
                L_tx = np.eye(num_tx)

        H_total_normalized = np.zeros((num_rx, num_tx), dtype=np.complex128)
        c = 3e8
        lambda_0 = c / self.config.center_frequency_hz

        # 步骤 9, 10, 11: 对所有簇进行循环，计算并叠加其贡献
        for cluster in clusters:
            # 生成IID瑞利衰落矩阵
            H_iid = (np.random.randn(num_rx, num_tx) + 1j * np.random.randn(num_rx, num_tx)) / np.sqrt(2)
            # 应用空间相关性
            H_corr = L_rx @ H_iid @ L_tx.conj().T
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

        return H_total_normalized