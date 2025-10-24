import numpy as np
import math
from .abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters
from channel_model_38901.parameters.parameter_loader import ParameterTables


class UMa_ChannelCondition(ABC_ChannelConditionModel):
    """UMa场景的信道条件模型实现。"""

    def get_condition(self, distance_2d: float, h_ut: float) -> bool:
        """
        计算UMa场景的LOS概率。
        参考TR 38.901 表 7.4.2-1。
        """
        if distance_2d <= 18:
            return True  # LOS

        # C(d2D, hUT) 计算
        if h_ut < 13:
            c_h_ut = 0
        elif 13 <= h_ut <= 23:
            g_d2d = (1.25e-6 * distance_2d ** 3) * np.exp(-distance_2d / 150) if distance_2d > 18 else 0
            c_h_ut = ((h_ut - 13) / 10) ** 1.5 * g_d2d
        else:  # h_ut > 23
            c_h_ut = 0

        prob_los = (18 / distance_2d + (1 - 18 / distance_2d) * np.exp(-distance_2d / 63)) * \
                   (1 + c_h_ut)

        return bool(np.random.rand() < prob_los)


class UMa_Pathloss(ABC_PathlossModel):
    """
    UMa场景的路径损耗模型实现。
    参考TR 38.901 表 7.4.1-1。
    """

    def __init__(self, params: ParameterTables):
        self.params = params.get_scenario_params('UMa')

    def calculate_pathloss(self, config: ScenarioConfig, is_los: bool) -> float:
        """计算UMa场景的路径损耗（dB）。"""
        fc_ghz = config.center_frequency_hz / 1e9
        h_bs = config.bs_positions[0, 2]
        h_ut = config.ue_positions[0, 2]

        d_2d = np.linalg.norm(config.bs_positions[0, :2] - config.ue_positions[0, :2])
        d_3d = np.linalg.norm(config.bs_positions[0] - config.ue_positions[0])

        h_e = 1.0
        h_bs_eff = h_bs - h_e
        h_ut_eff = h_ut - h_e

        d_bp = 4 * h_bs_eff * h_ut_eff * (config.center_frequency_hz / 3e8)

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


class UMa_Fading(ABC_FadingModel):
    """UMa场景的大尺度衰落模型实现。"""

    def __init__(self, params: ParameterTables):
        self.params = params.get_scenario_params('UMa')

    def generate_lsps(self, config: ScenarioConfig, is_los: bool, fc_ghz: float, use_median_lsps: bool = False) -> LargeScaleParameters:
        """为UMa场景生成大尺度参数。"""
        lsp_set = 'lsp_parameters_los' if is_los else 'lsp_parameters_nlos'
        lsp_params = self.params[lsp_set]
        correlations = lsp_params['correlations']

        param_order = ['DS', 'ASD', 'ASA', 'ZSA', 'ZSD','SF', 'K']
        k_is_na = lsp_params.get(f'mu_log10_K') == 'N/A'
        if k_is_na:
            param_order.remove('K')

        mu = []
        sigma = []

        d_2d = np.linalg.norm(config.bs_positions[0, :2] - config.ue_positions[0, :2])
        h_ut = config.ue_positions[0, 2]
        # 在 local_scope 中加入 d_2d 和 h_ut 以便 eval() 使用
        local_scope = {'log10': math.log10, 'fc_ghz': fc_ghz, 'd_2d': d_2d, 'h_ut': h_ut, 'max': max}

        for p in param_order:
            mu_str = lsp_params[f'mu_log10_{p}']
            sigma_str = lsp_params[f'sigma_log10_{p}']
            mu.append(eval(mu_str, {"__builtins__": {}}, local_scope))
            sigma.append(eval(sigma_str, {"__builtins__": {}}, local_scope))

        if use_median_lsps:
            x = np.array(mu)
        else:
            # 构建相关矩阵
            C = np.identity(len(param_order))
            for i, p1 in enumerate(param_order):
                for j, p2 in enumerate(param_order):
                    if i < j:
                        key = f'{p1}_vs_{p2}'
                        if key in correlations:
                            C[i, j] = C[j, i] = correlations[key]

            # Cholesky分解 - 对非正定矩阵进行特征值修正
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # 寻找最近的正定矩阵
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues[eigenvalues < 1e-12] = 1e-12  # 替换负特征值
                C_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                L = np.linalg.cholesky(C_new)

            # 生成相关的随机变量
            z = np.random.randn(len(param_order))
            y = L @ z

            # 应用均值和标准差
            x = np.array(mu) + y * np.array(sigma)

        # 创建LSP对象，转换对数正态参数
        lsp_dict = dict(zip(param_order, x))

        return LargeScaleParameters(
            delay_spread_s=10 ** lsp_dict['DS'],
            departure_azimuth_spread_deg=10 ** lsp_dict['ASD'],
            arrival_azimuth_spread_deg=10 ** lsp_dict['ASA'],
            arrival_zenith_spread_deg=10 ** lsp_dict['ZSA'],
            shadow_fade_db=lsp_dict['SF'],
            k_factor_linear=10 ** (lsp_dict['K'] / 10) if 'K' in lsp_dict else 0.0,
            # ZSD在标准中处理方式不同，通常由ZSA导出
            departure_zenith_spread_deg=10 ** lsp_dict['ZSD']  # 占位符
        )

    # ==================== 在这里新增一个方法 ====================
    def get_zod_offset(self, config: ScenarioConfig, is_los: bool) -> float:
        """
        为UMa场景实现ZOD（离开天顶角）平均偏移量的计算。
        该实现严格遵循 TR 38.901 V19.0.0 的 Table 7.5-7。
        """
        # 根据 Table 7.5-7，UMa LOS 条件下，偏移量为 0
        if is_los:
            return 0.0
        # 根据 Table 7.5-7，UMa NLOS 条件下，偏移量是一个复杂公式
        else:
            # 从config对象中获取计算所需的参数
            d_2d = np.linalg.norm(config.bs_positions[0, :2] - config.ue_positions[0, :2])
            h_ut = config.ue_positions[0, 2]
            fc_ghz = config.center_frequency_hz / 1e9

            # 实现表格 Note 中定义的 a, b, c, e 辅助函数
            # 这些是局部函数，只在该方法内部有效
            def a(fc):
                return 0.208 * math.log10(fc) - 0.782

            def b(fc):
                return 25

            def c(fc):
                return -0.13 * math.log10(fc) + 2.03

            def e(fc):
                return 7.66 * math.log10(fc) - 5.96

            # 应用 UMa NLOS ZOD offset 的完整公式
            offset = e(fc_ghz) * (10 ** (-a(fc_ghz) * math.log10(max(b(fc_ghz), d_2d)))) + c(fc_ghz) * (h_ut - 1.5)

            return offset
    # ==========================================================