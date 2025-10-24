import numpy as np
from abc import ABC, abstractmethod
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters

class ABC_ChannelConditionModel(ABC):
    """ 
    用于确定信道条件（LOS/NLOS）的抽象基类。
    """
    @abstractmethod
    def get_condition(self, distance_2d: float, h_ut: float) -> bool:
        """ 
        根据2D距离和UE高度，返回信道条件。

        Args:
            distance_2d (float): 2D距离。
            h_ut (float): UE高度。

        Returns:
            bool: LOS则返回True，NLOS则返回False。
        """
        pass

class ABC_PathlossModel(ABC):
    """ 
    路径损耗模型的抽象基类。
    """

    @abstractmethod
    def calculate_pathloss(self, config: ScenarioConfig, is_los: bool) -> float:
        """ 
        计算路径损耗（dB）。

        Args:
            config (ScenarioConfig): 场景配置。
            is_los (bool): 是否为LOS。

        Returns:
            float: 路径损耗（dB）。
        """
        pass

class ABC_FadingModel(ABC):
    """ 
    大尺度衰落模型的抽象基类。
    """

    @abstractmethod
    def generate_lsps(self, config: ScenarioConfig, is_los: bool, fc_ghz: float, use_median_lsps: bool = False) -> LargeScaleParameters:
        """ 
        生成一组大尺度参数。

        Args:
            is_los (bool): 是否为LOS。
            fc_ghz (float): 中心频率（GHz）。
            use_median_lsps (bool, optional): 是否使用LSP中位数。默认为False。

        Returns:
            LargeScaleParameters: 大尺度参数对象。
        """
        pass

    def get_zod_offset(self, config: ScenarioConfig, is_los: bool) -> float:
        """
        根据具体场景，计算ZOD（离开天顶角）的平均偏移量 mu_offset,ZOD。

        这是一个可以在子类中被重写的方法。
        基类中的默认实现返回0，适用于那些标准中没有定义特殊偏移的场景。

        Args:
            config (ScenarioConfig): 场景配置，用于获取距离、高度等信息。
            is_los (bool): 是否为LOS条件。

        Returns:
            float: 计算出的 ZOD 平均偏移量（单位：度）。
        """
        return 0.0
    # ==========================================================
