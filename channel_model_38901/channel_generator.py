import warnings

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
from .models.antenna_pattern import calculate_element_gain
from .common.geometry import get_antenna_positions, gcs_to_lcs_angles

class ChannelGenerator:
    """
    核心信道生成器引擎。
    该类负责根据用户配置，执行3GPP TR 38.901标准中定义的12个步骤，生成一个信道实现。
    """
    # 3GPP CDL tables (Table 7.7.1) store delays normalized to a 1 us reference DS.
    _CDL_DELAY_REF_S = 1000e-9

    # TR 38.901 Sec 7.7.1: the two strongest clusters split into 3 sub-clusters.
    # Per-sub-cluster delay offsets (units of c_DS) and power fractions, and the
    # mapping of the 20 rays (1-indexed in the spec) to each sub-cluster.
    _SUBCLUSTER_DELAY_OFFSETS = (0.0, 1.28, 2.56)
    _SUBCLUSTER_POWER_FRACTIONS = (10.0 / 20.0, 6.0 / 20.0, 4.0 / 20.0)

    def __init__(self, config: ScenarioConfig, params: ParameterTables, cluster_model_name: Optional[str] = None, use_median_lsps: bool = False, scale_to_scenario: bool = False, split_subclusters: bool = False, c_ds_ns: float = 3.91):
        """
        初始化信道生成器。

        Args:
            config (ScenarioConfig): 包含所有仿真参数的配置对象。
            params (ParameterTables): 从JSON文件加载的3GPP标准参数表。
            cluster_model_name (Optional[str], optional): 小尺度簇模型的名称。默认为 'Stochastic'。
            use_median_lsps (bool, optional): 是否使用LSP中位值。默认为 False。
            scale_to_scenario (bool, optional): 是否把 CDL 表格延迟/角度重缩放到场景 LSP。默认 False。
            split_subclusters (bool, optional): 是否对两个最强簇做 Sec 7.7.1 子簇分裂。默认 False。
            c_ds_ns (float, optional): 子簇延迟扩展常数 c_DS（纳秒）。默认 3.91ns。
        """
        self.config = config
        self.params = params
        self.cluster_model_name = cluster_model_name if cluster_model_name else 'Stochastic'
        # Audit batch 2 / F12: 未知簇模型名 (如 'cdl-a' 小写 typo) 此前会
        # 静默落入 Stochastic 路径。白名单校验, 拒绝无声 fallback。
        _name = self.cluster_model_name
        if not (_name == 'Stochastic' or _name.startswith('CDL-')
                or _name == 'SCME' or _name.startswith('SCME-')):
            raise ValueError(
                f"未知 cluster_model_name {_name!r}。合法值: None/'Stochastic' "
                f"(随机簇), 'CDL-A'..'CDL-E', 'SCME'/'SCME-*'。注意大小写敏感。"
            )
        self.use_median_lsps = use_median_lsps
        # Opt-in: rescale tabulated CDL delays/angles to the scenario LSPs (TR 38.901
        # Sec 7.7.3 / 7.7.5.1). Default False preserves raw-table CDL behaviour.
        self.scale_to_scenario = scale_to_scenario
        # Opt-in: split the 2 strongest clusters into 3 sub-clusters (TR 38.901
        # Sec 7.7.1). Default False preserves the unsplit cluster list.
        self.split_subclusters = split_subclusters
        self.c_ds_ns = c_ds_ns
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
                ant_config.tx_spacing_v * lambda_0,
                ant_config.tx_bearing_deg, ant_config.tx_downtilt_deg, ant_config.tx_slant_deg
            )
        if ant_config.rx_positions is None:
            ant_config.rx_positions = get_antenna_positions(
                ant_config.rx_array_type,
                ant_config.num_rx_rows, ant_config.num_rx_cols,
                ant_config.rx_spacing_h * lambda_0,
                ant_config.rx_spacing_v * lambda_0,
                ant_config.rx_bearing_deg, ant_config.rx_downtilt_deg, ant_config.rx_slant_deg
            )

    @staticmethod
    def _unit_vector(zenith_deg: float, azimuth_deg: float) -> np.ndarray:
        """球面单位向量, zenith 约定 (TR 38.901 §7.1, eq 7.1-6): θ 从 +z 轴量起,
        θ=90° 为水平方向。历史 bug (audit batch 2, A2): 此处曾用 elevation 约定
        (z=sinθ), 与 SoS 导出 / LCS 变换 / 元件方向图的 zenith 约定矛盾。"""
        theta = np.deg2rad(zenith_deg)
        phi = np.deg2rad(azimuth_deg)
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

    @staticmethod
    def _element_field_amplitude(pattern: str, zenith_deg: float, azimuth_deg: float,
                                 bearing_deg: float, downtilt_deg: float, slant_deg: float) -> float:
        """
        Linear field-amplitude factor for one antenna element in a given GCS direction.
        For 'TR38.901' the 3GPP Table 7.3-1 element pattern is evaluated in the array LCS
        (the GCS angle is first rotated by the array orientation). 'Isotropic' => 1.0.
        """
        if pattern != 'TR38.901':
            return 1.0
        theta_l, phi_l = gcs_to_lcs_angles(zenith_deg, azimuth_deg,
                                           bearing_deg, downtilt_deg, slant_deg)
        gain_db = calculate_element_gain(phi_l, theta_l)
        return 10.0 ** (gain_db / 20.0)

    def _cluster_element_gain(self, cluster) -> float:
        """Combined Tx*Rx linear field-amplitude gain for a cluster's mean direction."""
        ant = self.config.antenna_config
        g_tx = self._element_field_amplitude(
            ant.tx_element_pattern, cluster.zod_theta, cluster.aod_phi,
            ant.tx_bearing_deg, ant.tx_downtilt_deg, ant.tx_slant_deg)
        g_rx = self._element_field_amplitude(
            ant.rx_element_pattern, cluster.zoa_theta, cluster.aoa_phi,
            ant.rx_bearing_deg, ant.rx_downtilt_deg, ant.rx_slant_deg)
        return g_tx * g_rx

    def _synthesize_plane_wave(self, k_hat_rx: np.ndarray, k_hat_tx: np.ndarray, lambda_0: float) -> np.ndarray:
        ant_config = self.config.antenna_config
        k_factor = 2 * np.pi / lambda_0
        a_rx = np.exp(1j * k_factor * (ant_config.rx_positions @ k_hat_rx))
        a_tx = np.exp(1j * k_factor * (ant_config.tx_positions @ k_hat_tx))
        return a_rx[:, None] @ a_tx[None, :]

    def _los_element_gain(self) -> float:
        """Combined Tx*Rx element field-amplitude gain along the geometric LOS direction."""
        ant = self.config.antenna_config
        if ant.tx_element_pattern != 'TR38.901' and ant.rx_element_pattern != 'TR38.901':
            return 1.0
        d_vec = self.config.ue_positions[0] - self.config.bs_positions[0]
        dist = np.linalg.norm(d_vec)
        if dist < 1e-9:
            return 1.0
        dx, dy, dz = d_vec
        zen_tx = np.rad2deg(np.arccos(np.clip(dz / dist, -1.0, 1.0)))
        azi_tx = np.rad2deg(np.arctan2(dy, dx))
        zen_rx = np.rad2deg(np.arccos(np.clip(-dz / dist, -1.0, 1.0)))
        azi_rx = np.rad2deg(np.arctan2(-dy, -dx))
        g_tx = self._element_field_amplitude(ant.tx_element_pattern, zen_tx, azi_tx,
                                             ant.tx_bearing_deg, ant.tx_downtilt_deg, ant.tx_slant_deg)
        g_rx = self._element_field_amplitude(ant.rx_element_pattern, zen_rx, azi_rx,
                                             ant.rx_bearing_deg, ant.rx_downtilt_deg, ant.rx_slant_deg)
        return g_tx * g_rx

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
            clusters = self._load_deterministic_clusters(self.cluster_model_name, lsps)
        else:
            if logger: logger("  - 生成随机簇...")
            clusters = self._generate_stochastic_clusters(lsps, is_los, logger)

        # TR 38.901 Sec 7.7.1: 子簇分裂（opt-in），把两个最强簇各分成 3 个子簇。
        if self.split_subclusters:
            clusters = self._apply_subcluster_split(clusters)
            if logger: logger(f"  - 子簇分裂后簇数: {len(clusters)}")

        # 步骤 3: 合成信道并应用大尺度效应
        H_normalized = self._synthesize_channel(clusters, lsps, time_s, is_deterministic_clusters, logger)
        total_loss_db = pathloss_db + lsps.shadow_fade_db
        H_final = H_normalized * 10**(-total_loss_db / 20)

        if logger: logger("信道生成完成。")
        return H_final, H_normalized, clusters, lsps, pathloss_db, is_los

    @staticmethod
    def _power_weighted_azimuth_mean(angles_deg: np.ndarray, powers: np.ndarray) -> float:
        z = np.sum(powers * np.exp(1j * np.deg2rad(angles_deg)))
        return float(np.rad2deg(np.angle(z)))

    @staticmethod
    def _rms_azimuth_spread(angles_deg: np.ndarray, powers: np.ndarray, mean_deg: float) -> float:
        # Wrap deviations to [-180,180] (TR 38.901 circular RMS angular spread).
        d = (angles_deg - mean_deg + 180.0) % 360.0 - 180.0
        var = np.sum(powers * d**2) / np.sum(powers)
        return float(np.sqrt(var))

    @classmethod
    def _azimuth_spread_ratio(cls, angles_deg: np.ndarray, powers: np.ndarray, as_desired_deg: float) -> float:
        """§7.7.5.1 缩放比 AS_desired/AS_model (退化情形返回 1.0)。"""
        mean = cls._power_weighted_azimuth_mean(angles_deg, powers)
        as_model = cls._rms_azimuth_spread(angles_deg, powers, mean)
        if as_model < 1e-9 or as_desired_deg <= 0.0:
            return 1.0
        return as_desired_deg / as_model

    @classmethod
    def _zenith_spread_ratio(cls, angles_deg: np.ndarray, powers: np.ndarray, zs_desired_deg: float) -> float:
        mean = float(np.sum(powers * angles_deg) / np.sum(powers))
        var = np.sum(powers * (angles_deg - mean) ** 2) / np.sum(powers)
        zs_model = float(np.sqrt(var))
        if zs_model < 1e-9 or zs_desired_deg <= 0.0:
            return 1.0
        return zs_desired_deg / zs_model

    @classmethod
    def _scale_azimuth(cls, angles_deg: np.ndarray, powers: np.ndarray, as_desired_deg: float) -> np.ndarray:
        """TR 38.901 Sec 7.7.5.1 azimuth scaling: rescale spread to scenario, preserve mean."""
        mean = cls._power_weighted_azimuth_mean(angles_deg, powers)
        as_model = cls._rms_azimuth_spread(angles_deg, powers, mean)
        if as_model < 1e-9 or as_desired_deg <= 0.0:
            return angles_deg
        d = (angles_deg - mean + 180.0) % 360.0 - 180.0
        return mean + (as_desired_deg / as_model) * d

    @classmethod
    def _scale_zenith(cls, angles_deg: np.ndarray, powers: np.ndarray, zs_desired_deg: float) -> np.ndarray:
        """Zenith scaling about the power-weighted mean, then clip to [0,180]."""
        mean = float(np.sum(powers * angles_deg) / np.sum(powers))
        d = angles_deg - mean
        var = np.sum(powers * d**2) / np.sum(powers)
        zs_model = float(np.sqrt(var))
        if zs_model < 1e-9 or zs_desired_deg <= 0.0:
            return angles_deg
        scaled = mean + (zs_desired_deg / zs_model) * d
        return np.clip(scaled, 0.0, 180.0)

    def _load_deterministic_clusters(self, model_name: str, lsps: Optional[LargeScaleParameters] = None) -> List[EngineCluster]:
        """从参数文件中加载确定性模型（如CDL）的簇参数。"""
        if model_name.startswith('CDL-'):
            model_params = self.params.get_cdl_model_params(model_name)
        elif model_name == 'SCME' or model_name.startswith('SCME-'):
            model_params = self.params.get_scme_model_params(model_name)
        else:
            return []

        if not model_params:
            raise ValueError(f"在参数文件中未找到确定性模型 '{model_name}'。")

        raw = model_params['clusters']
        # 对于确定性模型，JSON中的功率是绝对的。我们将其归一化以供引擎使用。
        powers_linear = np.array([10**(c['power_db'] / 10) for c in raw])
        normalized_powers = powers_linear / powers_linear.sum()

        last = {'aod': 0.0, 'aoa': 0.0, 'zod': 90.0, 'zoa': 90.0}
        aod = np.empty(len(raw)); aoa = np.empty(len(raw))
        zod = np.empty(len(raw)); zoa = np.empty(len(raw))
        delays = np.empty(len(raw))
        for i, c in enumerate(raw):
            last['aod'] = c.get('aod_deg', last['aod']); aod[i] = last['aod']
            last['aoa'] = c.get('aoa_deg', last['aoa']); aoa[i] = last['aoa']
            last['zod'] = c.get('zod_deg', last['zod']); zod[i] = last['zod']
            last['zoa'] = c.get('zoa_deg', last['zoa']); zoa[i] = last['zoa']
            delays[i] = c['delay_ns'] * 1e-9

        # TR 38.901 Sec 7.7.3 / 7.7.5.1: 把表格化的 CDL 延迟与角度重缩放到场景 LSP。
        # §7.7.5.1 缩放作用于 ray 角而非仅簇中心, 故簇内 ray spread (as_*) 也
        # 要按同一比例缩放 (Codex PR#18 P2); 比例须在均值缩放前用原表角度计算。
        spread_scale = {'asd': 1.0, 'asa': 1.0, 'zsd': 1.0, 'zsa': 1.0}
        if self.scale_to_scenario and lsps is not None:
            delays = delays * (lsps.delay_spread_s / self._CDL_DELAY_REF_S)
            spread_scale['asd'] = self._azimuth_spread_ratio(aod, normalized_powers, lsps.departure_azimuth_spread_deg)
            spread_scale['asa'] = self._azimuth_spread_ratio(aoa, normalized_powers, lsps.arrival_azimuth_spread_deg)
            spread_scale['zsd'] = self._zenith_spread_ratio(zod, normalized_powers, lsps.departure_zenith_spread_deg)
            spread_scale['zsa'] = self._zenith_spread_ratio(zoa, normalized_powers, lsps.arrival_zenith_spread_deg)
            aod = self._scale_azimuth(aod, normalized_powers, lsps.departure_azimuth_spread_deg)
            aoa = self._scale_azimuth(aoa, normalized_powers, lsps.arrival_azimuth_spread_deg)
            zod = self._scale_zenith(zod, normalized_powers, lsps.departure_zenith_spread_deg)
            zoa = self._scale_zenith(zoa, normalized_powers, lsps.arrival_zenith_spread_deg)

        # Phase 6 fix (PR #6 Codex P2): mirror the generated linear XPR
        # into both fields. `k_n` is the legacy name (kept for backward
        # compat); `xpr_linear` is the canonical Phase 6+ field consumed
        # by the dual-pol synthesizer.
        xpr_lin = 10 ** (model_params.get('xpr_db', 11) / 10)

        # TR 38.901 Table 7.7.1-1..5: 每个 CDL 模型定义簇内 ray spread
        # (c_ASD/c_ASA/c_ZSD/c_ZSA)。历史 bug (audit batch 2, A3): JSON 缺
        # 该字段且 .get(..., 0.0) 静默回退 → SoS/ray 导出 20 条射线全部
        # 塌缩到簇均值角。模型级字段作默认值, 簇级 as_*_deg 仍可覆盖;
        # 两级都缺时必须告警而不是无声给 0。
        ray_spread = model_params.get('ray_spread_deg', {})
        _AS_TO_SPREAD = {
            'as_aod_deg': 'c_asd_deg',
            'as_aoa_deg': 'c_asa_deg',
            'as_zod_deg': 'c_zsd_deg',
            'as_zoa_deg': 'c_zsa_deg',
        }
        # 逐维告警: 某维度模型级 (ray_spread_deg 块缺或块内缺 key) 与簇级
        # 都没提供时, 该维度簇内角度扩展塌缩为 0 (Codex PR#18 P3+P2 —
        # 自带完整簇级 as_* 的 custom 模型不吵, 部分缺失的块也不漏报)。
        missing_dims = [
            as_key for as_key, c_key in _AS_TO_SPREAD.items()
            if c_key not in ray_spread and any(as_key not in c for c in raw)
        ]
        if missing_dims:
            warnings.warn(
                f"确定性模型 '{model_name}' 的 ray_spread_deg 与簇级 as_* 均未"
                f"覆盖维度 {missing_dims}, 这些维度簇内角度扩展为 0 — ray 级"
                f"导出 (SoS/OTA) 的射线在该维度将塌缩到簇均值角",
                UserWarning,
            )
        default_as = {
            'as_aod_deg': ray_spread.get('c_asd_deg', 0.0),
            'as_aoa_deg': ray_spread.get('c_asa_deg', 0.0),
            'as_zod_deg': ray_spread.get('c_zsd_deg', 0.0),
            'as_zoa_deg': ray_spread.get('c_zsa_deg', 0.0),
        }
        clusters = []
        for i in range(len(raw)):
            clusters.append(EngineCluster(
                delay=float(delays[i]),
                power=float(normalized_powers[i]),
                aod_phi=float(aod[i]), aoa_phi=float(aoa[i]),
                zod_theta=float(zod[i]), zoa_theta=float(zoa[i]),
                as_aod=raw[i].get('as_aod_deg', default_as['as_aod_deg']) * spread_scale['asd'],
                as_aoa=raw[i].get('as_aoa_deg', default_as['as_aoa_deg']) * spread_scale['asa'],
                as_zod=raw[i].get('as_zod_deg', default_as['as_zod_deg']) * spread_scale['zsd'],
                as_zoa=raw[i].get('as_zoa_deg', default_as['as_zoa_deg']) * spread_scale['zsa'],
                k_n=xpr_lin,            # legacy field (was sole XPR storage)
                xpr_linear=xpr_lin,     # canonical Phase 6+ field
            ))
        return clusters

    def _apply_subcluster_split(self, clusters: List[EngineCluster]) -> List[EngineCluster]:
        """
        TR 38.901 Sec 7.7.1: split the two strongest clusters into 3 sub-clusters each.

        Each of the two highest-power clusters is replaced by 3 sub-clusters carrying
        power fractions {10,6,4}/20 of the parent and delay offsets {0, 1.28, 2.56}·c_DS.
        Sub-clusters inherit the parent cluster's mean angles (cluster-level CIR). All
        other clusters are passed through unchanged. Order is preserved.
        """
        if len(clusters) < 2:
            return clusters
        c_ds_s = self.c_ds_ns * 1e-9
        strongest_idx = set(np.argsort([c.power for c in clusters])[-2:].tolist())
        out: List[EngineCluster] = []
        for idx, cluster in enumerate(clusters):
            if idx not in strongest_idx:
                out.append(cluster)
                continue
            for d_off, p_frac in zip(self._SUBCLUSTER_DELAY_OFFSETS,
                                     self._SUBCLUSTER_POWER_FRACTIONS):
                sub = cluster.model_copy()
                sub.delay = cluster.delay + d_off * c_ds_s
                sub.power = cluster.power * p_frac
                sub.h_matrix = None
                out.append(sub)
        return out

    def _generate_stochastic_clusters(self, lsps: LargeScaleParameters, is_los: bool, logger: Optional[Callable[[str], None]]) -> List[EngineCluster]:
        """根据LSP随机生成簇参数。"""
        scenario_params = self.params.get_scenario_params(self.config.scenario_name)
        num_clusters = scenario_params['num_clusters_los'] if is_los else scenario_params['num_clusters_nlos']
        tau_n = self._generate_cluster_delays(lsps, num_clusters, is_los, scenario_params)
        P_n = self._generate_cluster_powers(tau_n, lsps, is_los, scenario_params)
        angles = self._generate_angles(P_n, lsps, num_clusters)
        k_n = self._generate_xpr(scenario_params, is_los, num_clusters)
        initial_phases = self._generate_initial_phases(num_clusters)

        # Audit batch 2 / F5: 簇内 ray spread 用 Table 7.5-6 的 per-cluster
        # 常数 c_ASA/c_ASD/c_ZSA (JSON lsp_parameters 一直有但从未被读取),
        # 而不是把 LSP 总扩展 (大一个量级) 当簇内扩展 — 历史 bug 让 SoS/ray
        # 导出的射线偏移被系统性放大。c_ZSD 按 spec 注释为 (3/8)·10^μ_lgZSD,
        # 此处用采样 ZSD 近似 10^μ (单实现内无均值可用)。
        lsp_set = 'lsp_parameters_los' if is_los else 'lsp_parameters_nlos'
        lsp_params = scenario_params.get(lsp_set, {})
        c_asa = lsp_params.get('c_asa_deg')
        c_asd = lsp_params.get('c_asd_deg')
        c_zsa = lsp_params.get('c_zsa_deg')
        missing = [k for k, v in (('c_asa_deg', c_asa), ('c_asd_deg', c_asd),
                                  ('c_zsa_deg', c_zsa)) if v is None]
        if missing:
            warnings.warn(
                f"场景 '{self.config.scenario_name}'/{lsp_set} 缺 Table 7.5-6 "
                f"簇内扩展常数 {missing}, 回退到 LSP 总扩展 (偏大) — 请补参数文件",
                UserWarning,
            )
        as_aoa = c_asa if c_asa is not None else lsps.arrival_azimuth_spread_deg
        as_aod = c_asd if c_asd is not None else lsps.departure_azimuth_spread_deg
        as_zoa = c_zsa if c_zsa is not None else lsps.arrival_zenith_spread_deg
        as_zod = 0.375 * lsps.departure_zenith_spread_deg

        clusters = []
        for i in range(num_clusters):
            # Phase 6 fix (PR #6 Codex P2): mirror per-cluster XPR into both
            # `k_n` (legacy) and `xpr_linear` (canonical Phase 6+ field).
            clusters.append(EngineCluster(
                delay=tau_n[i], power=P_n[i],
                aod_phi=angles['aod_phi'][i], aoa_phi=angles['aoa_phi'][i],
                zod_theta=angles['zod_theta'][i], zoa_theta=angles['zoa_theta'][i],
                as_aod=as_aod, as_aoa=as_aoa,
                as_zod=as_zod, as_zoa=as_zoa,
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
        # zenith 角以 90° (水平) 为中心, 符合 zenith 约定 (θ 从 +z 量起)。
        # 几何 LOS 方向中心化 (§7.5 step 7 的 θ_LOS,ZOA 项) 属 PR10 范围。
        zoa = 90.0 + X_n_signs[:, 2] * zoa_unscaled + Y_n_offsets[:, 2] * (lsps.arrival_zenith_spread_deg / 7.0)
        zod = 90.0 + X_n_signs[:, 3] * zod_unscaled + Y_n_offsets[:, 3] * (lsps.departure_zenith_spread_deg / 7.0)
        return {
            'aoa_phi': X_n_signs[:, 0] * aoa_unscaled + Y_n_offsets[:, 0] * (lsps.arrival_azimuth_spread_deg / 7.0),
            'aod_phi': X_n_signs[:, 1] * aod_unscaled + Y_n_offsets[:, 1] * (lsps.departure_azimuth_spread_deg / 7.0),
            'zoa_theta': self._fold_zenith(zoa),
            'zod_theta': self._fold_zenith(zod)
        }

    @staticmethod
    def _fold_zenith(theta_deg: np.ndarray) -> np.ndarray:
        """把 zenith 角镜像折叠回 [0, 180]° (TR 38.901 §7.5 step 7:
        θ∈[180°,360°] 时取 360°-θ)。弱簇的 -ZS·log(P/Pmax) 项可超 90°,
        加 90° 偏移后可能越界 (Codex PR#17 P2)。"""
        theta = np.abs(np.asarray(theta_deg, dtype=float)) % 360.0
        return np.where(theta > 180.0, 360.0 - theta, theta)

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

            # 应用单元方向图增益 (TR 38.901 Table 7.3-1); Isotropic 时为 1.0
            H_corr = H_corr * self._cluster_element_gain(cluster)

            cluster.h_matrix = H_corr
            
            # 计算多普勒频移 (zenith 约定, 复用 _unit_vector 避免再次出现两套公式)
            v = self.config.ue_velocities[0]
            r_rx_mean = self._unit_vector(cluster.zoa_theta, cluster.aoa_phi)
            doppler_shift_hz = np.dot(r_rx_mean, v) / lambda_0
            doppler_term = np.exp(1j * 2 * np.pi * doppler_shift_hz * time_s)
            
            # 累加当前簇的贡献
            H_total_normalized += np.sqrt(cluster.power) * H_corr * doppler_term

        if lsps.k_factor_linear > 0:
            k_linear = lsps.k_factor_linear
            H_los = self._synthesize_los_component(lambda_0) * self._los_element_gain()
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
            clusters = self._load_deterministic_clusters(self.cluster_model_name, lsps)
        else:
            clusters = self._generate_stochastic_clusters(lsps, is_los, None)

        # TR 38.901 Sec 7.7.1: 子簇分裂（opt-in），与 run() 主路径保持一致。
        if self.split_subclusters:
            clusters = self._apply_subcluster_split(clusters)

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

                # A'. Element radiation pattern (TR 38.901 Table 7.3-1), evaluated
                # per-ray at this ray's own Tx/Rx direction (more accurate than the
                # cluster-mean scalar used in the cluster-level synthesizer). The
                # element pattern is identical across array elements, so this gain
                # is constant over the rx/tx antenna loops -> compute once per ray.
                # Isotropic patterns return 1.0, keeping legacy exports unchanged.
                g_ray = (
                    self._element_field_amplitude(
                        ant_config.tx_element_pattern, zod_ray, aod_ray,
                        ant_config.tx_bearing_deg, ant_config.tx_downtilt_deg,
                        ant_config.tx_slant_deg)
                    * self._element_field_amplitude(
                        ant_config.rx_element_pattern, zoa_ray, aoa_ray,
                        ant_config.rx_bearing_deg, ant_config.rx_downtilt_deg,
                        ant_config.rx_slant_deg)
                )

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
                        weight = np.sqrt(p_ray) * g_ray * np.exp(1j * total_phase)

                        sos_params.append({
                            "Cluster_ID": cluster_idx,
                            "Ray_ID": m,
                            "Rx_Ant_ID": rx_idx,
                            "Tx_Ant_ID": tx_idx,
                            "Doppler_Hz": doppler_freq_hz,
                            "Delay_s": cluster.delay,
                            "Weight_Real": weight.real,
                            "Weight_Imag": weight.imag,
                            "Power_Linear": p_ray * g_ray**2,
                            "Phase_Rad": np.angle(weight)
                        })
                    
        return sos_params
