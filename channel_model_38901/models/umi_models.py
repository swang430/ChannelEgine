
import numpy as np
import math
from .abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters
from channel_model_38901.parameters.parameter_loader import ParameterTables

class UMi_ChannelCondition(ABC_ChannelConditionModel):
    """Implementation of the UMi-StreetCanyon channel condition model."""
    def get_condition(self, distance_2d: float, h_ut: float) -> bool:
        if distance_2d <= 18:
            return True
        prob_los = 18 / distance_2d + (1 - 18 / distance_2d) * np.exp(-distance_2d / 36)
        return np.random.rand() < prob_los

class UMi_Pathloss(ABC_PathlossModel):
    """Implementation of the UMi-StreetCanyon pathloss model."""
    def __init__(self, params: ParameterTables):
        self.params = params.get_scenario_params('UMi-StreetCanyon')

    def calculate_pathloss(self, config: ScenarioConfig, is_los: bool) -> float:
        fc_ghz = config.center_frequency_hz / 1e9
        h_bs = config.bs_positions[0, 2]
        h_ut = config.ue_positions[0, 2]
        d_3d = np.linalg.norm(config.bs_positions[0] - config.ue_positions[0])
        d_bp = 4 * (h_bs - 1) * (h_ut - 1) * (config.center_frequency_hz / 3e8)

        local_scope = {
            'log10': math.log10,
            'd_3d': d_3d,
            'fc_ghz': fc_ghz,
            'd_bp': d_bp,
            'h_bs': h_bs,
            'h_ut': h_ut,
            'max': max,
            'np': np
        }

        if is_los:
            los_params = self.params['pathloss_los']
            formula_1_str = los_params['formula_1']
            formula_2_str = los_params['formula_2']
            pl1 = eval(formula_1_str, {"__builtins__": {}}, local_scope)
            pl2 = eval(formula_2_str, {"__builtins__": {}}, local_scope)
            pathloss = max(pl1, pl2)
        else:
            los_params = self.params['pathloss_los']
            formula_1_str = los_params['formula_1']
            formula_2_str = los_params['formula_2']
            pl1_los = eval(formula_1_str, {"__builtins__": {}}, local_scope)
            pl2_los = eval(formula_2_str, {"__builtins__": {}}, local_scope)
            pathloss_los = max(pl1_los, pl2_los)
            
            nlos_params = self.params['pathloss_nlos']
            formula_nlos_str = nlos_params['formula']
            local_scope['pathloss_los'] = pathloss_los
            pathloss = eval(formula_nlos_str, {"__builtins__": {}}, local_scope)
            
        return pathloss

class UMi_Fading(ABC_FadingModel):
    """Implementation of the UMi-StreetCanyon large-scale fading model."""
    def __init__(self, params: ParameterTables):
        self.params = params.get_scenario_params('UMi-StreetCanyon')

    def generate_lsps(self, config: ScenarioConfig, is_los: bool, fc_ghz: float, use_median_lsps: bool = False) -> LargeScaleParameters:
        lsp_set = 'lsp_parameters_los' if is_los else 'lsp_parameters_nlos'
        # UMi has the same LSP parameters for LOS and NLOS in the provided JSON
        lsp_params = self.params['lsp_parameters_los'] 
        correlations = lsp_params['correlations']

        param_order = ['DS', 'ASD', 'ASA', 'ZSA', 'ZSD', 'SF', 'K']
        k_is_na = lsp_params.get(f'mu_log10_K') == 'N/A'
        if k_is_na:
            param_order.remove('K')

        mu = []
        sigma = []
        d_2d = np.linalg.norm(config.bs_positions[0, :2] - config.ue_positions[0, :2])
        h_ut = config.ue_positions[0, 2]
        h_bs = config.bs_positions[0, 2]
        # 在 local_scope 中加入 d_2d 和 h_ut 以便 eval() 使用
        local_scope = {'log10': math.log10, 'fc_ghz': fc_ghz, 'd_2d': d_2d, 'h_ut': h_ut,'h_bs': h_bs, 'max': max, 'abs': abs}

        for p in param_order:
            mu_str = lsp_params[f'mu_log10_{p}']
            sigma_str = lsp_params[f'sigma_log10_{p}']
            mu.append(eval(mu_str, {"__builtins__": {}}, local_scope))
            sigma.append(eval(sigma_str, {"__builtins__": {}}, local_scope))

        if use_median_lsps:
            x = np.array(mu)
        else:
            C = np.identity(len(param_order))
            for i, p1 in enumerate(param_order):
                for j, p2 in enumerate(param_order):
                    if i < j:
                        key = f'{p1}_vs_{p2}'
                        if key in correlations:
                            C[i, j] = C[j, i] = correlations[key]

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues[eigenvalues < 1e-12] = 1e-12
                C_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                L = np.linalg.cholesky(C_new)
            
            z = np.random.randn(len(param_order))
            y = L @ z
            x = np.array(mu) + y * np.array(sigma)
        
        lsp_dict = dict(zip(param_order, x))

        return LargeScaleParameters(
            delay_spread_s=10**lsp_dict['DS'],
            departure_azimuth_spread_deg=10**lsp_dict['ASD'],
            arrival_azimuth_spread_deg=10**lsp_dict['ASA'],
            arrival_zenith_spread_deg=10**lsp_dict['ZSA'],
            shadow_fade_db=lsp_dict['SF'],
            k_factor_linear=10**(lsp_dict['K']/10) if 'K' in lsp_dict else 0.0,
            departure_zenith_spread_deg=10**lsp_dict['ZSD']
        )

    # ==================== 在这里新增一个方法 ====================
    def get_zod_offset(self, config: ScenarioConfig, is_los: bool) -> float:
        """
        为UMi-StreetCanyon场景实现ZOD（离开天顶角）平均偏移量的计算。
        该实现严格遵循 TR 38.901 V19.0.0 的 Table 7.5-8。
        """
        # 根据 Table 7.5-8，UMi-Street Canyon LOS 条件下，偏移量为 0
        if is_los:
            return 0.0
        # 根据 Table 7.5-8，NLOS 条件下，偏移量是一个与2D距离相关的函数
        else:
            # 从config对象中获取计算所需的2D距离
            d_2d = np.linalg.norm(config.bs_positions[0, :2] - config.ue_positions[0, :2])

            # 应用 UMi-Street Canyon NLOS ZOD offset 的完整公式
            offset = -10 ** (-1.5 * math.log10(max(10, d_2d))) + 3.3

            return offset
    # ==========================================================