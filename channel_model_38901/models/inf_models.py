import numpy as np
from .abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters
from channel_model_38901.common.expression_eval import safe_eval
from channel_model_38901.parameters.parameter_loader import ParameterTables

class InF_ChannelCondition(ABC_ChannelConditionModel):
    """Implementation of the InF channel condition model."""
    def get_condition(self, distance_2d: float, h_ut: float) -> bool:
        if distance_2d <= 5:
            return True
        elif distance_2d <= 49:
            prob_los = np.exp(-(distance_2d - 5) / 70.8)
        else:
            prob_los = 0.54 * np.exp(-(distance_2d - 49) / 211.7)
        return np.random.rand() < prob_los

class InF_Pathloss(ABC_PathlossModel):
    """Implementation of the InF pathloss model."""
    def __init__(self, params: ParameterTables):
        self.params = params.get_scenario_params('InF')

    def calculate_pathloss(self, config: ScenarioConfig, is_los: bool) -> float:
        fc_ghz = config.center_frequency_hz / 1e9
        d_3d = np.linalg.norm(config.bs_positions[0] - config.ue_positions[0])

        local_scope = {
            'd_3d': d_3d,
            'fc_ghz': fc_ghz,
        }

        if is_los:
            los_params = self.params['pathloss_los']
            formula_1_str = los_params['formula_1']
            pathloss = safe_eval(formula_1_str, local_scope)
        else:
            los_params = self.params['pathloss_los']
            formula_1_str = los_params['formula_1']
            pathloss_los = safe_eval(formula_1_str, local_scope)
            
            nlos_params = self.params['pathloss_nlos']
            formula_nlos_str = nlos_params['formula']
            local_scope['pathloss_los'] = pathloss_los
            pathloss = safe_eval(formula_nlos_str, local_scope)
            
        return pathloss

class InF_Fading(ABC_FadingModel):
    """Implementation of the InF large-scale fading model."""
    def __init__(self, params: ParameterTables):
        self.params = params.get_scenario_params('InF')

    def generate_lsps(self, config: ScenarioConfig, is_los: bool, use_median_lsps: bool = False) -> LargeScaleParameters:
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
            'V': self.params['V'],
            'S': self.params['S'],
        }

        for p in param_order:
            mu_str = lsp_params[f'mu_log10_{p}']
            sigma_str = lsp_params[f'sigma_log10_{p}']
            mu.append(safe_eval(mu_str, local_scope))
            sigma.append(safe_eval(sigma_str, local_scope))

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
            departure_zenith_spread_deg=10**(zsd_log10 if zsd_log10 is not None else lsp_dict['ZSA'])
        )
