
from abc import ABC, abstractmethod
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters

class ABC_ChannelConditionModel(ABC):
    """Abstract Base Class for determining the channel condition (LOS/NLOS)."""
    @abstractmethod
    def get_condition(self, distance_2d: float, h_ut: float) -> bool:
        """Returns True for LOS, False for NLOS."""
        pass

class ABC_PathlossModel(ABC):
    """Abstract Base Class for pathloss models."""

    @abstractmethod
    def calculate_pathloss(self, config: ScenarioConfig, is_los: bool) -> float:
        """Calculates the pathloss in dB."""
        pass

class ABC_FadingModel(ABC):
    """Abstract Base Class for large-scale fading models."""

    @abstractmethod
    def generate_lsps(self, is_los: bool) -> LargeScaleParameters:
        """Generates a set of large-scale parameters."""
        pass
