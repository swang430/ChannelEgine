import numpy as np
from .abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters
from channel_model_38901.common.expression_eval import safe_eval
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
            g_d2d = (1.25e-6 * distance_2d**3) * np.exp(-distance_2d / 150) if distance_2d > 18 else 0
            c_h_ut = ((h_ut - 13) / 10)**1.5 * g_d2d
        else: # h_ut > 23
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
            'd_3d': d_3d,
            'fc_ghz': fc_ghz,
            'd_bp': d_bp,
            'h_bs': h_bs,
            'h_ut': h_ut,
        }

        if is_los:
            los_params = self.params['pathloss_los']
            formula_1_str = los_params['formula_1']
            formula_2_str = los_params['formula_2']
            
            pl1 = safe_eval(formula_1_str, local_scope)
            pl2 = safe_eval(formula_2_str, local_scope)
            
            pathloss = max(pl1, pl2)
        else:
            los_params = self.params['pathloss_los']
            formula_1_str = los_params['formula_1']
            formula_2_str = los_params['formula_2']
            pl1_los = safe_eval(formula_1_str, local_scope)
            pl2_los = safe_eval(formula_2_str, local_scope)
            pathloss_los = max(pl1_los, pl2_los)
            
            nlos_params = self.params['pathloss_nlos']
            formula_nlos_str = nlos_params['formula']
            local_scope['pathloss_los'] = pathloss_los
            pathloss = safe_eval(formula_nlos_str, local_scope)
            
        return pathloss

class UMa_Fading(ABC_FadingModel):
    """UMa场景的大尺度衰落模型实现。"""
    def __init__(self, params: ParameterTables):
        self.params = params.get_scenario_params('UMa')

    def generate_lsps(self, config: ScenarioConfig, is_los: bool, use_median_lsps: bool = False) -> LargeScaleParameters:
        """为UMa场景生成大尺度参数。"""
        lsp_set = 'lsp_parameters_los' if is_los else 'lsp_parameters_nlos'
        lsp_params = self.params[lsp_set]
        correlations = lsp_params['correlations']

        param_order = ['DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K']
        k_is_na = lsp_params.get(f'mu_log10_K') == 'N/A'
        if k_is_na:
            param_order.remove('K')

        mu = []
        sigma = []
        
        fc_ghz = config.center_frequency_hz / 1e9
        h_ut = config.ue_positions[0, 2]
        h_bs = config.bs_positions[0, 2]
        d_2d = np.linalg.norm(config.bs_positions[0, :2] - config.ue_positions[0, :2])

        local_scope = {
            'fc_ghz': fc_ghz,
            'h_ut': h_ut,
            'h_bs': h_bs,
            'd_2d': d_2d,
        }

        for p in param_order:
            mu_str = lsp_params[f'mu_log10_{p}']
            sigma_str = lsp_params[f'sigma_log10_{p}']
            mu.append(safe_eval(mu_str, local_scope))
            sigma.append(safe_eval(sigma_str, local_scope))

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
                eigenvalues[eigenvalues < 1e-12] = 1e-12 # 替换负特征值
                C_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                L = np.linalg.cholesky(C_new)
            
            # 生成相关的随机变量
            z = np.random.randn(len(param_order))
            y = L @ z
            
            # 应用均值和标准差
            x = np.array(mu) + y * np.array(sigma)

        # 创建LSP对象，转换对数正态参数
        lsp_dict = dict(zip(param_order, x))
        zsd_mu_expr = lsp_params.get('mu_log10_ZSD')
        zsd_sigma_expr = lsp_params.get('sigma_log10_ZSD')
        zsd_log10 = None
        if zsd_mu_expr and zsd_sigma_expr and zsd_mu_expr != 'N/A':
            zsd_mu = safe_eval(zsd_mu_expr, local_scope)
            zsd_sigma = safe_eval(zsd_sigma_expr, local_scope)
            if use_median_lsps:
                zsd_log10 = zsd_mu
            else:
                zsd_log10 = zsd_mu + np.random.randn() * zsd_sigma

        return LargeScaleParameters(
            delay_spread_s=10**lsp_dict['DS'],
            departure_azimuth_spread_deg=10**lsp_dict['ASD'],
            arrival_azimuth_spread_deg=10**lsp_dict['ASA'],
            arrival_zenith_spread_deg=10**lsp_dict['ZSA'],
            shadow_fade_db=lsp_dict['SF'],
            k_factor_linear=10**(lsp_dict['K']/10) if 'K' in lsp_dict else 0.0,
            # ZSD在标准中处理方式不同，通常由ZSA导出
            departure_zenith_spread_deg=10**(zsd_log10 if zsd_log10 is not None else lsp_dict['ZSA'])
        )
