from typing import Optional, Callable, Dict
from pathlib import Path
from .common.data_models import ScenarioConfig, EngineCluster, LargeScaleParameters
from .parameters.parameter_loader import ParameterLoader
from .channel_generator import ChannelGenerator
from .common.geometry import get_antenna_positions
import numpy as np

class ChannelSimulator:
    """
    外观类（Facade Class），为运行信道仿真提供一个简洁的顶层API。
    这是用户与信道模型库交互的主要入口点。
    """
    def __init__(self, scenario_name: str, center_frequency_hz: float, cluster_model_name: Optional[str] = None, use_median_lsps: bool = False, scale_to_scenario: bool = False, split_subclusters: bool = False, c_ds_ns: float = 3.91):
        """
        初始化信道仿真器。

        Args:
            scenario_name (str): 大尺度环境的名称，例如 'UMa', 'RMa'。这决定了路径损耗和LSP的统计特性。
            center_frequency_hz (float): 载波中心频率，单位为赫兹。
            cluster_model_name (Optional[str], optional): 小尺度簇模型的名称，例如 'CDL-A', 'SCME-UMa' 或 'Stochastic'。
                                                       如果为 None 或 'Stochastic'，将根据大尺度环境随机生成簇。
                                                       默认为 None。
            use_median_lsps (bool, optional): 是否使用大尺度参数（LSP）的中位值，而不是从统计分布中随机抽样。
                                             这对于需要确定性LSP的测试非常有用。默认为 False。
            scale_to_scenario (bool, optional): 是否把 CDL 表格延迟/角度/簇内扩展重缩放到场景 LSP
                                             (TR 38.901 §7.7.3 / §7.7.5.1)。默认 False (原表值)。
            split_subclusters (bool, optional): 是否对两个最强簇做 §7.7.1 子簇分裂。默认 False。
            c_ds_ns (float, optional): 子簇延迟扩展常数 c_DS (纳秒), 仅 split_subclusters=True 时生效。
                                             默认 3.91 ns。

        Note:
            BS/UE 位置默认 (0,0,25)/(100,0,1.5), 速度默认 0 — 用
            set_bs_ue_positions() / set_velocities() 覆盖 (影响 pathloss、
            LSP 表达式中的 d_2d/h_*、LOS 方向与多普勒)。
        """
        param_file = Path(__file__).resolve().parent / "parameters" / "parameters1013_tr38901_v19.0.0.json"
        self.params = ParameterLoader.load_parameters(str(param_file))
        
        self.config = ScenarioConfig(
            scenario_name=scenario_name,
            center_frequency_hz=center_frequency_hz,
            cluster_model_name=cluster_model_name,
            bs_positions=np.array([[0, 0, 25]]),
            ue_positions=np.array([[100, 0, 1.5]]),
            ue_velocities=np.array([[0, 0, 0]])
        )
        
        self.scale_to_scenario = scale_to_scenario
        self.split_subclusters = split_subclusters
        self.c_ds_ns = c_ds_ns
        self._use_median_lsps = use_median_lsps
        self.generator = self._build_generator()
        self._last_results = None
        self._last_lsps = None

    def _build_generator(self) -> ChannelGenerator:
        """Construct a ChannelGenerator carrying the current option flags."""
        return ChannelGenerator(
            self.config, self.params, self.config.cluster_model_name,
            use_median_lsps=self._use_median_lsps,
            scale_to_scenario=self.scale_to_scenario,
            split_subclusters=self.split_subclusters,
            c_ds_ns=self.c_ds_ns,
        )

    def _invalidate_cache(self):
        self._last_results = None
        self._last_lsps = None

    def run(self, force_condition: Optional[str] = None, time_s: float = 0.0, logger: Optional[Callable[[str], None]] = print, use_median_lsps: Optional[bool] = None) -> Dict:
        """
        运行一次信道实现（channel realization）并返回包含结果的字典。

        Args:
            force_condition (Optional[str], optional): 强制指定信道条件为 'LOS' (视距) 或 'NLOS' (非视距)。
                                                     如果为 None，则根据标准中的概率模型自动确定。默认为 None。
            time_s (float, optional): 仿真时间点，单位为秒。用于计算多普勒效应。默认为 0.0。
            logger (Optional[Callable[[str], None]], optional): 用于记录仿真过程日志的回调函数。默认为 print。
            use_median_lsps (Optional[bool], optional): 在单次运行中覆盖初始化时设置的 use_median_lsps 选项。

        Returns:
            Dict: 包含仿真结果的字典，键包括:
                'channel_matrix': 最终的信道矩阵 (H_final)。
                'channel_matrix_normalized': 归一化的信道矩阵 (H_normalized)。
                'clusters': 包含簇信息的列表 (List[EngineCluster])。
                'lsps': 生成的大尺度参数 (LargeScaleParameters)。
                'pathloss_db': 计算出的路径损耗，单位为dB。
                'is_los': 布尔值，指示当前信道是否为LOS。
        """
        if logger:
            logger(f"Running simulation for environment: {self.config.scenario_name} with cluster model: {self.config.cluster_model_name} at time t={time_s}s")
        
        H_final, H_normalized, clusters, lsps, pathloss_db, is_los = self.generator.generate_channel_realization(
            force_condition=force_condition, time_s=time_s, logger=logger, use_median_lsps=use_median_lsps
        )
        
        results = {
            'channel_matrix': H_final,
            'channel_matrix_normalized': H_normalized,
            'clusters': clusters,
            'lsps': lsps,
            'pathloss_db': pathloss_db,
            'is_los': is_los
        }
        self._last_results = results
        self._last_lsps = lsps
        return results

    def set_bs_ue_positions(self, bs_positions: np.ndarray, ue_positions: np.ndarray):
        """更新基站 (BS) 和用户设备 (UE) 的位置。"""
        self.config.bs_positions = bs_positions
        self.config.ue_positions = ue_positions
        self.generator = self._build_generator()
        self._invalidate_cache()

    def set_velocities(self, ue_velocities: np.ndarray):
        """更新用户设备 (UE) 的速度。"""
        self.config.ue_velocities = ue_velocities
        self.generator = self._build_generator()
        self._invalidate_cache()

    def set_antenna_config(self, num_tx_rows, num_tx_cols, num_rx_rows, num_rx_cols,
                           tx_spacing_h=0.5, tx_spacing_v=0.5, rx_spacing_h=0.5, rx_spacing_v=0.5,
                           tx_array_type='ULA', rx_array_type='ULA',
                           tx_element_pattern=None, rx_element_pattern=None,
                           tx_bearing_deg=None, tx_downtilt_deg=None, tx_slant_deg=None,
                           rx_bearing_deg=None, rx_downtilt_deg=None, rx_slant_deg=None):
        """更新天线阵列配置。

        element_pattern / orientation 参数为 None 时保留 antenna_config 现有值
        （默认 Isotropic / 0°），从而对既有调用方保持向后兼容。
        """
        ac = self.config.antenna_config
        ac.num_tx_rows = num_tx_rows
        ac.num_tx_cols = num_tx_cols
        ac.num_rx_rows = num_rx_rows
        ac.num_rx_cols = num_rx_cols
        ac.tx_spacing_h = tx_spacing_h
        ac.tx_spacing_v = tx_spacing_v
        ac.rx_spacing_h = rx_spacing_h
        ac.rx_spacing_v = rx_spacing_v
        ac.tx_array_type = tx_array_type
        ac.rx_array_type = rx_array_type
        if tx_element_pattern is not None: ac.tx_element_pattern = tx_element_pattern
        if rx_element_pattern is not None: ac.rx_element_pattern = rx_element_pattern
        if tx_bearing_deg is not None: ac.tx_bearing_deg = tx_bearing_deg
        if tx_downtilt_deg is not None: ac.tx_downtilt_deg = tx_downtilt_deg
        if tx_slant_deg is not None: ac.tx_slant_deg = tx_slant_deg
        if rx_bearing_deg is not None: ac.rx_bearing_deg = rx_bearing_deg
        if rx_downtilt_deg is not None: ac.rx_downtilt_deg = rx_downtilt_deg
        if rx_slant_deg is not None: ac.rx_slant_deg = rx_slant_deg

        # Calculate positions
        c = 3e8
        lam = c / self.config.center_frequency_hz
        
        # Tx
        self.config.antenna_config.tx_positions = get_antenna_positions(
            self.config.antenna_config.tx_array_type,
            num_tx_rows, num_tx_cols,
            self.config.antenna_config.tx_spacing_h * lam,
            self.config.antenna_config.tx_spacing_v * lam,
            self.config.antenna_config.tx_bearing_deg,
            self.config.antenna_config.tx_downtilt_deg,
            self.config.antenna_config.tx_slant_deg
        )

        # Rx
        self.config.antenna_config.rx_positions = get_antenna_positions(
            self.config.antenna_config.rx_array_type,
            num_rx_rows, num_rx_cols,
            self.config.antenna_config.rx_spacing_h * lam,
            self.config.antenna_config.rx_spacing_v * lam,
            self.config.antenna_config.rx_bearing_deg,
            self.config.antenna_config.rx_downtilt_deg,
            self.config.antenna_config.rx_slant_deg
        )
        
        self.generator = self._build_generator()
        self._invalidate_cache()

    def set_scenario(self, scenario_name: str, cluster_model_name: Optional[str] = None):
        """更新仿真场景和簇模型。"""
        self.config.scenario_name = scenario_name
        self.config.cluster_model_name = cluster_model_name
        self.generator = self._build_generator()
        self._invalidate_cache()
        
    def set_median_lsp_usage(self, use_median_lsps: bool):
        """设置是否使用LSP中位值。"""
        self._use_median_lsps = use_median_lsps
        self.generator = self._build_generator()
        self._invalidate_cache()

    def get_config(self) -> ScenarioConfig:
        """返回当前的仿真配置对象。"""
        return self.config

    def get_lsps(self) -> Optional[LargeScaleParameters]:
        """获取最近一次生成的大尺度参数。"""
        if self._last_lsps is None:
            _, _, _, lsps, _, _ = self.generator.generate_channel_realization(logger=None)
            self._last_lsps = lsps
        return self._last_lsps

    def export_fpga_sos_params(self):
        """
        导出用于FPGA SoS的参数列表。
        """
        return self.generator.export_sos_parameters()
