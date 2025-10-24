import numpy as np
import math
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

    def __init__(self, config: ScenarioConfig, params: ParameterTables, cluster_model_name: Optional[str] = None,
                 use_median_lsps: bool = False):
        # --- 对应 3GPP Step 1: Set environment (部分) ---
        # 存储场景配置 (ScenarioConfig 包含地点, 频率, 天线等)
        self.config = config
        self.params = params# 加载的参数表
        self.cluster_model_name = cluster_model_name if cluster_model_name else 'Stochastic'# 确定性或随机簇
        self.use_median_lsps = use_median_lsps# 是否使用LSP中值
        self._select_models()# 根据场景选择模型

    def _select_models(self):
        # --- 对应 3GPP Step 1: Set environment (部分) ---
        # 根据 config 中的场景名称选择对应的 LOS/NLOS 判断、路径损耗和 LSP 生成模型
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
            self.condition_model = cond_model()# LOS/NLOS 模型
            self.pathloss_model = pl_model(self.params)# 路径损耗模型
            self.fading_model = fade_model(self.params)# 大尺度参数模型
        else:
            raise NotImplementedError(f"场景 '{scenario}' 的LSP生成暂不支持。")

    def _calculate_geometric_los_angles(self) -> dict:
        # --- 辅助计算 ---用于 Step 7
        # 根据 BS 和 UE 的位置计算几何视距 (LOS) 路径的角度 (AOD, ZOD, AOA, ZOA)
        bs_pos = self.config.bs_positions[0]
        ue_pos = self.config.ue_positions[0]
        vec = ue_pos - bs_pos
        dx, dy, dz = vec[0], vec[1], vec[2]
        d_3d = np.linalg.norm(vec)
        if d_3d == 0:
            return {'aod': 0, 'zod': 90, 'aoa': 180, 'zoa': 90}
        los_aod = np.rad2deg(np.arctan2(dy, dx))
        los_zod = np.rad2deg(np.arccos(dz / d_3d))
        los_aoa = np.rad2deg(np.arctan2(-dy, -dx))
        los_zoa = np.rad2deg(np.arccos(-dz / d_3d))
        return {'aod': los_aod, 'zod': los_zod, 'aoa': los_aoa, 'zoa': los_zoa}

    # 信道实现
    def generate_channel_realization(self, force_condition: Optional[str] = None, time_s: float = 0.0,
                                     logger: Optional[Callable[[str], None]] = print,
                                     use_median_lsps: Optional[bool] = None) -> tuple:
        """
        生成单次的信道实现（最终版，处理双极化输出）。
        严格执行 7.5-28
        """
        los_angles = self._calculate_geometric_los_angles()
        if logger: logger(
            f"  - 几何LOS角度: AOD={los_angles['aod']:.2f}, ZOD={los_angles['zod']:.2f}, AOA={los_angles['aoa']:.2f}, ZOA={los_angles['zoa']:.2f}")

        if logger: logger(f"开始为环境 '{self.config.scenario_name}' 生成信道...")

        # --- 对应 3GPP Step 2: Determine LOS/NLOS state ---
        h_ut = self.config.ue_positions[0, 2]
        d_2d = np.linalg.norm(self.config.bs_positions[0, :2] - self.config.ue_positions[0, :2])
        is_los = self.condition_model.get_condition(d_2d, h_ut)# 根据模型概率判断
        if force_condition == 'LOS': is_los = True# 用户强制指定
        if force_condition == 'NLOS': is_los = False
        # --- 对应 3GPP Step 3: Calculate pathloss ---
        # 计算基础路径损耗 PL (dB)
        pathloss_db = self.pathloss_model.calculate_pathloss(self.config, is_los)
        # --- 对应 3GPP Step 4: Generate large scale parameters (LSPs) ---
        # 生成包括阴影衰落(SF), K因子(KF), 时延扩展(DS), 角度扩展(AS)等
        fc_ghz = self.config.center_frequency_hz / 1e9
        use_median = use_median_lsps if use_median_lsps is not None else self.use_median_lsps
        lsps = self.fading_model.generate_lsps(self.config,is_los, fc_ghz, use_median_lsps=use_median)

        if logger:
            logger(f"  - 大尺度环境: {self.config.scenario_name}")
            logger(f"  - 簇模型: {self.cluster_model_name}")
            logger(f"  - 信道条件: {'LOS' if is_los else 'NLOS'}")
            logger(f"  - 路径损耗: {pathloss_db:.2f} dB")
            logger(f"  - LSP生成: {'使用中位值' if use_median else '使用随机采样'}")

        # STEP5-9生成簇参数 (Delays, Powers, Angles, XPR, Phases) ---
        is_deterministic_clusters = self.cluster_model_name.startswith('CDL-') or self.cluster_model_name.startswith(
            'SCME')
        if is_deterministic_clusters:
            # 加载确定性模型的簇参数 (CDL/SCME)
            clusters = self._load_deterministic_clusters(self.cluster_model_name)
            if logger: logger(f"  - 加载确定性簇参数: {self.cluster_model_name}")
        else:
            # 随机生成簇参数
            if logger: logger("  - 开始随机生成簇参数 ...")
            clusters = self._generate_stochastic_clusters(lsps, is_los, los_angles, logger)

        # step10-12: 合成信道并应用大尺度效应
        # _synthesize_channel
        H_normalized_matrix = self._synthesize_channel(clusters, lsps, time_s, is_deterministic_clusters, is_los,
                                                       logger)

        total_loss_db = pathloss_db + lsps.shadow_fade_db
        loss_linear = 10 ** (-total_loss_db / 20.0)

        # 应用大尺度损耗
        H_final_matrix = H_normalized_matrix * loss_linear

        if logger: logger("信道生成完成。")

        # --- [修改 : GUI 兼容性补丁] ---
        # 为了让您的 GUI 不崩溃，我们将最终的组合矩阵放入

        H_final_dict_for_gui = {
            'vv': H_final_matrix,
            'vh': np.zeros_like(H_final_matrix),
            'hv': np.zeros_like(H_final_matrix),
            'hh': np.zeros_like(H_final_matrix)
        }

        # H_normalized_dict 也做同样处理
        H_normalized_dict_for_gui = {
            'vv': H_normalized_matrix,
            'vh': np.zeros_like(H_normalized_matrix),
            'hv': np.zeros_like(H_normalized_matrix),
            'hh': np.zeros_like(H_normalized_matrix)
        }

        # 返回调整后的值
        return H_final_dict_for_gui, H_normalized_dict_for_gui, clusters, lsps, pathloss_db, is_los

    # --- [以下函数保持不变] ---

    def _load_deterministic_clusters(self, model_name: str) -> List[EngineCluster]:
        if model_name.startswith('CDL-'):
            model_params = self.params.get_cdl_model_params(model_name)
        elif model_name == 'SCME' or model_name.startswith('SCME-'):
            model_params = self.params.get_scme_model_params(model_name)
        else:
            return []
        if not model_params:
            raise ValueError(f"在参数文件中未找到确定性模型 '{model_name}'。")

        clusters = []
        total_power_linear = sum(10 ** (c['power_db'] / 10) for c in model_params['clusters'])
        last_aod, last_aoa, last_zod, last_zoa = 0, 0, 90, 90
        for c_data in model_params['clusters']:
            power_linear = 10 ** (c_data['power_db'] / 10)
            normalized_power = power_linear / total_power_linear
            aod = c_data.get('aod_deg', last_aod);
            aoa = c_data.get('aoa_deg', last_aoa)
            zod = c_data.get('zod_deg', last_zod);
            zoa = c_data.get('zoa_deg', last_zoa)
            last_aod, last_aoa, last_zod, last_zoa = aod, aoa, zod, zoa
            k_n_linear_float = 10 ** (model_params.get('xpr_db', 11) / 10)

            # 2. 将这个浮点数包装成一个 Numpy 数组 (包含一个元素)
            k_n_array = np.array([k_n_linear_float])
            clusters.append(EngineCluster(
                delay=c_data['delay_ns'] * 1e-9, power=normalized_power,
                aod_phi=aod, aoa_phi=aoa, zod_theta=zod, zoa_theta=zoa,
                as_aod=c_data.get('as_aod_deg', 0.0), as_aoa=c_data.get('as_aoa_deg', 0.0),
                as_zod=c_data.get('as_zod_deg', 0.0), as_zoa=c_data.get('as_zoa_deg', 0.0),
                k_n=k_n_array,
            ))
        return clusters

    def _generate_stochastic_clusters(self, lsps: LargeScaleParameters, is_los: bool, los_angles: dict,
                                      logger: Optional[Callable[[str], None]]) -> List[EngineCluster]:
        """根据LSP随机生成簇参数，现在包含射线级的XPR和相位。"""
        scenario_params = self.params.get_scenario_params(self.config.scenario_name)
        num_clusters_initial = scenario_params['num_clusters_los'] if is_los else scenario_params['num_clusters_nlos']
        num_rays = 20

        # --- 生成所有初始参数 ---
        # 对应 Step 5: Generate cluster delays
        tau_n = self._generate_cluster_delays(lsps, num_clusters_initial, is_los, scenario_params)
        # 对应 Step 6: Generate cluster powers
        P_n = self._generate_cluster_powers(tau_n, lsps, is_los, scenario_params)
        # 对应 Step 7: Generate arrival and departure angles
        angles = self._generate_angles(P_n, lsps, num_clusters_initial, is_los, los_angles, logger)
        # 对应 Step 9: Generate XPR
        k_n_rays = self._generate_xpr(scenario_params, is_los, num_clusters_initial, num_rays)
        # 对应 Step 10: Generate initial random phases
        initial_phases_rays = self._generate_initial_phases(num_clusters_initial, num_rays)

        # LOS时延补偿step5
        if is_los:
            k_factor_db = 10 * np.log10(lsps.k_factor_linear) if lsps.k_factor_linear > 0 else 0.0
            K = k_factor_db
            C_tau = 0.7705 - 0.0433 * K + 0.0002 * K ** 2 + 0.000017 * K ** 3
            tau_n = tau_n / C_tau

        # 移除弱簇 (Cluster Culling)step6
        max_power_linear = np.max(P_n)
        if max_power_linear > 0:
            threshold_linear = 10 ** ((10 * np.log10(max_power_linear) - 25) / 10)
            indices_to_keep = P_n >= threshold_linear

            num_before = len(P_n)
            tau_n = tau_n[indices_to_keep]
            P_n = P_n[indices_to_keep]
            k_n_rays = k_n_rays[indices_to_keep, :]
            initial_phases_rays = initial_phases_rays[indices_to_keep, :, :]
            for key in angles:
                angles[key] = angles[key][indices_to_keep]

        num_clusters_final = len(P_n)
        clusters = []
        for i in range(num_clusters_final):
            clusters.append(EngineCluster(
                delay=tau_n[i], power=P_n[i],
                aod_phi=angles['aod_phi'][i], aoa_phi=angles['aoa_phi'][i],
                zod_theta=angles['zod_theta'][i], zoa_theta=angles['zoa_theta'][i],
                k_n=k_n_rays[i, :],
                initial_phases=initial_phases_rays[i, :, :]
            ))
        return clusters

    def _generate_cluster_delays(self, lsps, num_clusters, is_los, scenario_params):
        # 实现step5根据 LSP 中的时延扩展 (DS) 生成簇时延 tau_n
        r_tau = scenario_params['delay_scaling_factor']
        X_n = np.random.rand(num_clusters)
        tau_prime = -r_tau * lsps.delay_spread_s * np.log(X_n)
        return np.sort(tau_prime - np.min(tau_prime))

    def _generate_cluster_powers(self, tau_n, lsps, is_los, scenario_params):
        # --- 实现 3GPP Step 6 ---
        # 根据簇时延 tau_n 和 LSP (DS, K-factor) 生成簇功率 P_n
        r_tau = scenario_params['delay_scaling_factor']
        P_prime = np.exp(-tau_n * (r_tau - 1) / (r_tau * lsps.delay_spread_s)) * 10 ** (
                -np.random.randn(len(tau_n)) / 10)
        if not is_los:
            return P_prime / np.sum(P_prime)
        else:
            K_R = lsps.k_factor_linear
            relative_scatter_powers = P_prime / np.sum(P_prime)
            scaled_scatter_powers = relative_scatter_powers * (1 / (K_R + 1))
            scaled_scatter_powers[0] += K_R / (K_R + 1)
            return scaled_scatter_powers

    def _generate_angles(self, P_n, lsps: LargeScaleParameters, num_clusters: int, is_los: bool,
                         los_angles: dict, logger=None):
        # --- 实现 3GPP Step 7 ---
        # 根据 LSP 中的角度扩展 (ASA, ASD, ZSA, ZSD) 和几何LOS角度生成簇角度
        scenario_params = self.params.get_scenario_params(self.config.scenario_name)
        k_factor_db = 10 * np.log10(lsps.k_factor_linear) if is_los and lsps.k_factor_linear > 0 else 0.0
        # ---  Step 7 Table7.5-2 ---
        c_phi_nlos_table = scenario_params.get('scaling_factors_aoa_aod', {})
        c_phi_nlos = c_phi_nlos_table.get(str(num_clusters), 1.273)
        # ---  Step 7 Table7.5-4 ---
        c_theta_nlos_table = scenario_params.get('scaling_factors_zoa_zod', {})
        c_theta_nlos = c_theta_nlos_table.get(str(num_clusters), 1.184)
        # ---  Step 7 7.5-10 ---
        if is_los:
            K = k_factor_db
            c_phi = c_phi_nlos * (1.1035 - 0.028 * K - 0.002 * K ** 2 + 0.0001 * K ** 3)
            c_theta = c_theta_nlos * (1.3086 + 0.0339 * K - 0.0077 * K ** 2 + 0.0002 * K ** 3)
        else:
            c_phi = c_phi_nlos
            c_theta = c_theta_nlos
        # ---  Step 7 7.5-9 ---
        P_n_safe = np.maximum(P_n, 1e-20)
        log_power_ratio = np.log(P_n_safe / np.max(P_n_safe))
        # ---  Step 7 7.5-9 ---
        phi_prime_aoa = (2 * lsps.arrival_azimuth_spread_deg / 1.4) * np.sqrt(-log_power_ratio) / c_phi
        phi_prime_aod = (2 * lsps.departure_azimuth_spread_deg / 1.4) * np.sqrt(-log_power_ratio) / c_phi
        # ---  Step 7 7.5-15 ---
        theta_prime_zoa = -lsps.arrival_zenith_spread_deg * log_power_ratio / c_theta
        theta_prime_zod = -lsps.departure_zenith_spread_deg * log_power_ratio / c_theta

        mu_offset_zod = self.fading_model.get_zod_offset(self.config, is_los)
        # ---  Step 7 7.5-11 Xn、Yn ---
        X_n_signs = np.random.choice([-1, 1], size=(num_clusters, 4))
        Y_n_offsets_aoa = np.random.randn(num_clusters) * (lsps.arrival_azimuth_spread_deg / 7.0)
        Y_n_offsets_aod = np.random.randn(num_clusters) * (lsps.departure_azimuth_spread_deg / 7.0)
        Y_n_offsets_zoa = np.random.randn(num_clusters) * (lsps.arrival_zenith_spread_deg / 7.0)
        Y_n_offsets_zod = np.random.randn(num_clusters) * (lsps.departure_zenith_spread_deg / 7.0)

        final_aoa = X_n_signs[:, 0] * phi_prime_aoa + Y_n_offsets_aoa + los_angles['aoa']
        final_aod = X_n_signs[:, 1] * phi_prime_aod + Y_n_offsets_aod + los_angles['aod']
        final_zoa = X_n_signs[:, 2] * theta_prime_zoa + Y_n_offsets_zoa + los_angles['zoa']
        final_zod = X_n_signs[:, 3] * theta_prime_zod + Y_n_offsets_zod + los_angles['zod'] + mu_offset_zod
        # ---  Step 7 7.5-12 、7.5-17、7.5-19---
        if is_los and num_clusters > 0:
            final_aoa[0] = los_angles['aoa']
            final_aod[0] = los_angles['aod']
            final_zoa[0] = los_angles['zoa']
            final_zod[0] = los_angles['zod'] + mu_offset_zod

        return {'aoa_phi': final_aoa, 'aod_phi': final_aod, 'zoa_theta': final_zoa, 'zod_theta': final_zod}

    def _generate_xpr(self, scenario_params, is_los, num_clusters, num_rays):
        # ---  Step 9 7.5-21  ---
        xpr_mu = scenario_params['xpr_mu_los'] if is_los else scenario_params['xpr_mu_nlos']
        xpr_sigma = scenario_params['xpr_sigma_los'] if is_los else scenario_params['xpr_sigma_nlos']
        X_nm = np.random.randn(num_clusters, num_rays) * xpr_sigma + xpr_mu
        return 10 ** (X_nm / 10)

    def _generate_initial_phases(self, num_clusters, num_rays):
        return np.random.uniform(-np.pi, np.pi, size=(num_clusters, num_rays, 4))

    # --- [新函数 1] ---
    def _calculate_array_response_from_positions(
            self,
            ant_positions: np.ndarray,
            zoa_rad: float,
            aoa_rad: float,
            lambda_0: float
    ) -> np.ndarray:
        """
        根据天线单元的物理3D坐标计算阵列响应向量。
        """
        if ant_positions is None or ant_positions.size == 0:
            num_ant = 1
            if ant_positions is None or ant_positions.size == 0:
                return np.array([1.0], dtype=np.complex128)
        else:
            num_ant = ant_positions.shape[0]

        r_hat_x = np.sin(zoa_rad) * np.cos(aoa_rad)
        r_hat_y = np.sin(zoa_rad) * np.sin(aoa_rad)
        r_hat_z = np.cos(zoa_rad)
        r_hat = np.array([r_hat_x, r_hat_y, r_hat_z])

        dot_products = ant_positions.dot(r_hat)
        k = (2 * np.pi) / lambda_0
        phase_shifts = k * dot_products
        response_vector = np.exp(1j * phase_shifts)

        return response_vector / np.sqrt(num_ant)

    # --- [新函数 2] ---
    def _get_element_field_pattern(self, pattern_name: str, angle_zod_rad: float, angle_aod_rad: float):
        """
        根据 3GPP TR 38.901 (Table 7.1-11) 计算天线单元的场方向图增益。
        """

        # --- 默认 Isotropic (各向同性) ---
        if pattern_name == 'Isotropic':
            return 1.0, 1.0

            # --- 将角度转换为规范所用的格式 ---
        theta_deg = np.rad2deg(angle_zod_rad)  # 对应 𝜃''
        phi_deg = np.rad2deg(angle_aod_rad)  # 对应 𝜙''

        phi_deg = (phi_deg + 180) % 360 - 180

        # --- 根据选择的模型设置参数 7.3-1---
        if pattern_name == '38901BS':
            # --- Table 7.3-1: 基站 (BS) 天线 ---
            theta_3dB = 65.0  # (度)
            phi_3dB = 65.0  # (度)
            SLA_V = 30.0  # (dB)
            A_max = 30.0  # (dB)
            G_E_max = 8.0  # (dBi)

        else:
            # 如果不是 '38901BS' ，则默认为 'Isotropic'
            return 1.0, 1.0

        # --- 实现 3GPP 公式 ---

        A_V_dB = -min(12 * ((theta_deg - 90.0) / theta_3dB) ** 2, SLA_V)
        A_H_dB = -min(12 * (phi_deg / phi_3dB) ** 2, A_max)
        A_E_dB = -min(-(A_V_dB + A_H_dB), A_max)
        G_E_dBi = G_E_max + A_E_dB
        gain_linear = 10 ** (G_E_dBi / 20.0)

        # --- 分配到极化分量 ---
        # 根据 3GPP (7.3.2) 的 Note，对于纯垂直极化 (zeta=0):
        F_theta_gain_linear = gain_linear  # F_theta = sqrt(A'')
        F_phi_gain_linear = 0.0  # F_phi = 0

        return F_theta_gain_linear, F_phi_gain_linear

    # --- [核心修改：严格执行 7.5-28 的 _synthesize_channel] ---
    def _synthesize_channel(self, clusters: List[EngineCluster], lsps: LargeScaleParameters, time_s: float,
                            is_deterministic: bool, is_los: bool,
                            logger: Optional[Callable[[str], None]]) -> np.ndarray:
        """
        严格按照 3GPP (7.5-28) 合成信道。
        【已修改】此函数现在返回一个组合后的 N_rx x N_tx 矩阵。
        """
        if logger: logger("  - 开始执行信道系数的合成 ...")

        # --- 1. 初始化和获取参数 ---
        ant_config = self.config.antenna_config
        tx_positions = ant_config.tx_positions
        rx_positions = ant_config.rx_positions

        num_tx = tx_positions.shape[0] if tx_positions is not None and tx_positions.size > 0 else 1
        num_rx = rx_positions.shape[0] if rx_positions is not None and rx_positions.size > 0 else 1

        # H 不再是一个字典，而是一个单独的矩阵
        H_total = np.zeros((num_rx, num_tx), dtype=np.complex128)

        c = 3e8
        lambda_0 = c / self.config.center_frequency_hz
        fc_hz = self.config.center_frequency_hz

        if is_deterministic:
            num_rays = 1
        else:
            scenario_params = self.params.get_scenario_params(self.config.scenario_name)
            ray_offsets = np.array(scenario_params.get('ray_angle_offsets_deg', np.zeros(20)))
            num_rays = len(ray_offsets)
            lsp_set = 'lsp_parameters_los' if is_los else 'lsp_parameters_nlos'
            lsp_params = scenario_params[lsp_set]
            c_asa_deg = lsp_params.get('c_asa_deg', lsps.arrival_azimuth_spread_deg)
            c_asd_deg = lsp_params.get('c_asd_deg', lsps.departure_azimuth_spread_deg)
            c_zsa_deg = lsp_params.get('c_zsa_deg', lsps.arrival_zenith_spread_deg)
            c_ds_ns = lsp_params.get('c_ds_ns', 0)

            cluster_powers = np.array([c.power for c in clusters])
            strongest_cluster_indices = np.argsort(cluster_powers)[-2:] if len(cluster_powers) > 1 else np.array([])
            # ---  Step 7 7.5-26---
            subcluster_delay_offsets = np.array([0, 1.28 * c_ds_ns * 1e-9, 2.56 * c_ds_ns * 1e-9])
            ray_to_subcluster_map = [0] * 8 + [1] * 4 + [2] * 4 + [1] * 2 + [0] * 2

        for n, cluster in enumerate(clusters):
            loop_range = 1 if is_deterministic else num_rays

            if not is_deterministic:
                # ---  Step 7 7.5-13---
                ray_aoa_set = cluster.aoa_phi + c_asa_deg * ray_offsets
                ray_aod_set = cluster.aod_phi + c_asd_deg * ray_offsets
                # ---  Step 7 7.5-18---
                ray_zoa_set = cluster.zoa_theta + c_zsa_deg * ray_offsets
                mu_log10_zsd_str = lsp_params.get('mu_log10_ZSD', lsp_params.get('mu_log10_ZSA', '0.0'))
                eval_scope = {'log10': math.log10, 'fc_ghz': fc_hz / 1e9,
                              'd_2d': np.linalg.norm(self.config.bs_positions[0, :2] - self.config.ue_positions[0, :2]),
                              'h_ut': self.config.ue_positions[0, 2], 'h_bs': self.config.bs_positions[0, 2],
                              'max': max, 'abs': abs}
                mu_log10_zsd = eval(mu_log10_zsd_str, {"__builtins__": {}}, eval_scope)
                offset_term = (3 / 8) * (10 ** mu_log10_zsd) * ray_offsets
                # ---  Step 7 7.5-20---
                ray_zod_set = cluster.zod_theta + offset_term
                # 对应 Step 8
                np.random.shuffle(ray_aoa_set)
                np.random.shuffle(ray_zoa_set)

            for m in range(loop_range):
                # 1. 获取射线参数 (功率, 角度, 延迟)
                if is_deterministic:
                    ray_power = cluster.power;
                    kappa = cluster.k_n if isinstance(cluster.k_n, (int, float)) else cluster.k_n[m]
                    ray_aoa, ray_aod = cluster.aoa_phi, cluster.aod_phi
                    ray_zoa, ray_zod = cluster.zoa_theta, cluster.zod_theta
                    ray_delay = cluster.delay
                    phases = np.random.uniform(-np.pi, np.pi, size=4)
                else:
                    ray_power = cluster.power / num_rays
                    kappa = cluster.k_n[m]
                    phases = cluster.initial_phases[m]
                    ray_aoa, ray_aod = ray_aoa_set[m], ray_aod_set[m]
                    ray_zoa, ray_zod = ray_zoa_set[m], ray_zod_set[m]
                    ray_delay = cluster.delay
                    if n in strongest_cluster_indices:
                        subcluster_idx = ray_to_subcluster_map[m]
                        ray_delay += subcluster_delay_offsets[subcluster_idx]
                is_main_los_cluster = is_los and n == 0

                # 2. 计算信道散射矩阵 (公式中的 2x2 矩阵)
                if is_main_los_cluster:
                    F_vv, F_vh, F_hv, F_hh = 1.0, 0.0, 0.0, -1.0
                else:
                    if kappa <= 0: kappa = 1e12  # 防止除以零
                    F_vv = np.exp(1j * phases[0])  # 对应 VV (theta -> theta)
                    F_vh = np.sqrt(1 / kappa) * np.exp(1j * phases[1])  # 对应 VH (phi -> theta)
                    F_hv = np.sqrt(1 / kappa) * np.exp(1j * phases[2])  # 对应 HV (theta -> phi)
                    F_hh = np.exp(1j * phases[3])  # 对应 HH (phi -> phi)

                # 3. 计算空间和时域项 (公式中的 exp() 项)
                ray_zoa_rad = np.deg2rad(ray_zoa)
                ray_aoa_rad = np.deg2rad(ray_aoa)
                ray_zod_rad = np.deg2rad(ray_zod)
                ray_aod_rad = np.deg2rad(ray_aod)

                # 阵列响应 (Term 5 & 6)
                a_rx = self._calculate_array_response_from_positions(
                    rx_positions, ray_zoa_rad, ray_aoa_rad, lambda_0
                )
                a_tx = self._calculate_array_response_from_positions(
                    tx_positions, ray_zod_rad, ray_aod_rad, lambda_0
                )

                # N_rx x N_tx 空间矩阵
                H_spatial = np.outer(a_rx, a_tx.conj())

                # 多普勒 (Term 7)
                v = self.config.ue_velocities[0]
                r_rx_mean = np.array([np.sin(ray_zoa_rad) * np.cos(ray_aoa_rad),
                                      np.sin(ray_zoa_rad) * np.sin(ray_aoa_rad),
                                      np.cos(ray_zoa_rad)])
                doppler_shift_hz = np.dot(r_rx_mean, v) / lambda_0
                doppler_term = np.exp(1j * 2 * np.pi * doppler_shift_hz * time_s)

                # 时延 (隐含项)
                delay_phase_term = np.exp(-1j * 2 * np.pi * fc_hz * ray_delay)

                # 射线功率 (Term 1: sqrt(P_n/M))
                power_term = np.sqrt(ray_power)

                # 4. 计算天线场方向图 (公式中的 F_rx 和 F_tx 向量)
                F_rx_theta, F_rx_phi = self._get_element_field_pattern(
                    ant_config.rx_element_pattern, ray_zoa_rad, ray_aoa_rad
                )
                F_tx_theta, F_tx_phi = self._get_element_field_pattern(
                    ant_config.tx_element_pattern, ray_zod_rad, ray_aod_rad
                )

                # 5. 【核心】严格执行 (7.5-28) 的 1x2 * 2x2 * 2x1 矩阵乘法
                # H_pol_total = [F_rx_theta, F_rx_phi] @ [[F_vv, F_vh], [F_hv, F_hh]] @ [F_tx_theta, F_tx_phi]

                # 手动展开矩阵乘法，得到一个标量:
                pol_scalar = (F_rx_theta * F_vv * F_tx_theta) + \
                             (F_rx_theta * F_vh * F_tx_phi) + \
                             (F_rx_phi * F_hv * F_tx_theta) + \
                             (F_rx_phi * F_hh * F_tx_phi)

                # 6. 组合所有项
                H_ray_contribution = power_term * pol_scalar * H_spatial * doppler_term * delay_phase_term

                # 7. 累加到总矩阵
                H_total += H_ray_contribution

        # 返回单个组合矩阵
        return H_total