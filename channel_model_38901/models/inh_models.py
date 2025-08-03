
from .abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters

class InH_ChannelCondition(ABC_ChannelConditionModel):
    def get_condition(self, distance_2d: float, h_ut: float) -> bool:
        # TODO: Implement InH LOS probability
        return super().get_condition(distance_2d, h_ut)

class InH_Pathloss(ABC_PathlossModel):
    def calculate_pathloss(self, config: ScenarioConfig, is_los: bool) -> float:
        # TODO: Implement InH pathloss
        return super().calculate_pathloss(config, is_los)

class InH_Fading(ABC_FadingModel):
    def generate_lsps(self, is_los: bool) -> LargeScaleParameters:
        # TODO: Implement InH LSP generation
        return super().generate_lsps(is_los)
