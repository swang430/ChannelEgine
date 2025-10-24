import numpy as np
import math
from .abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters
from channel_model_38901.parameters.parameter_loader import ParameterTables

class InH_ChannelCondition(ABC_ChannelConditionModel):
    """InH-Office 场景的信道条件模型实现。"""
    def get_condition(self, distance_2d: float, h_ut: float) -> bool:
        """
        计算 InH-Office 场景的 LOS 概率。
        参考 TR 38.901 表 7.4.2-1。这是一个分段函数。
        """
        if distance_2d <= 1.2:
            # 距离小于等于 1.2 米总是 LOS
            return True
        elif distance_2d <= 6.5:
            # 距离在 1.2 米 和 6.5 米之间
            prob_los = (1.2 / distance_2d) * (1 - np.exp(-distance_2d / 4.7)) + np.exp(-distance_2d / 4.7)
        else: # distance_2d > 6.5
            # 距离大于 6.5 米
            prob_los = 0.32 * np.exp(-(distance_2d - 6.5) / 32.6)
        # 根据概率进行随机判断
        return bool(np.random.rand() < prob_los)

class InH_Pathloss(ABC_PathlossModel):
    """InH-Office 场景的路径损耗模型实现。"""
    def __init__(self, params: ParameterTables):
        # 从参数加载器获取 InH-Office 场景的特定参数
        self.params = params.get_scenario_params('InH-Office')

    def calculate_pathloss(self, config: ScenarioConfig, is_los: bool) -> float:
        """
        计算 InH-Office 场景的路径损耗（dB）。
        参考 TR 38.901 表 7.4.1-1。
        """
        fc_ghz = config.center_frequency_hz / 1e9 # 载频 (GHz)
        # 计算 3D 距离
        d_3d = np.linalg.norm(config.bs_positions[0] - config.ue_positions[0])

        # 定义 eval() 函数可用的局部变量
        local_scope = {
            'log10': math.log10,
            'd_3d': d_3d,
            'fc_ghz': fc_ghz,
            'max': max,
            'np': np # 允许使用 numpy 函数
        }

        if is_los:
            # --- LOS 路径损耗计算 ---
            los_params = self.params['pathloss_los']
            formula_1_str = los_params['formula_1']
            # 直接计算 LOS 路径损耗
            pathloss = eval(formula_1_str, {"__builtins__": {}}, local_scope)
        else:
            # --- NLOS 路径损耗计算 ---
            # 首先计算 LOS 路径损耗 (NLOS 公式需要用到)
            los_params = self.params['pathloss_los']
            formula_1_str = los_params['formula_1']
            pathloss_los = eval(formula_1_str, {"__builtins__": {}}, local_scope)

            # 获取 NLOS 路径损耗参数
            nlos_params = self.params['pathloss_nlos']
            formula_nlos_str = nlos_params['formula']
            # 将计算出的 pathloss_los 添加到局部作用域
            local_scope['pathloss_los'] = pathloss_los
            # 计算 NLOS 路径损耗，并确保其不小于 LOS 路径损耗
            pathloss_nlos_calculated = eval(formula_nlos_str, {"__builtins__": {}}, local_scope)
            pathloss = max(pathloss_los, pathloss_nlos_calculated)

        return pathloss

class InH_Fading(ABC_FadingModel):
    """InH-Office 场景的大尺度衰落模型实现。"""
    def __init__(self, params: ParameterTables):
        # 从参数加载器获取 InH-Office 场景的特定参数
        self.params = params.get_scenario_params('InH-Office')

    # --- [已修改：添加 config 参数] ---
    def generate_lsps(self, config: ScenarioConfig, is_los: bool, fc_ghz: float, use_median_lsps: bool = False) -> LargeScaleParameters:
        """
        为 InH-Office 场景生成大尺度参数。
        参考 TR 38.901 Section 7.5 和 Table 7.5-6。
        """
        # 根据 LOS/NLOS 选择参数集
        lsp_set = 'lsp_parameters_los' if is_los else 'lsp_parameters_nlos'
        lsp_params = self.params[lsp_set]
        correlations = lsp_params['correlations']

        # --- [已修改：添加 'ZSD'] ---
        param_order = ['DS', 'ASD', 'ASA', 'ZSA', 'ZSD', 'SF', 'K']
        k_is_na = lsp_params.get(f'mu_log10_K') == 'N/A' # 检查 K 因子是否适用 (NLOS 不适用)
        if k_is_na:
            param_order.remove('K')

        mu = []
        sigma = []

        # --- [已修改：添加 d_2d, h_ut, h_bs, max, abs 到 local_scope] ---
        # 获取 eval() 可能需要的几何参数
        d_2d = np.linalg.norm(config.bs_positions[0, :2] - config.ue_positions[0, :2])
        h_ut = config.ue_positions[0, 2]
        h_bs = config.bs_positions[0, 2]
        local_scope = {
            'log10': math.log10,
            'fc_ghz': fc_ghz,
            'd_2d': d_2d,
            'h_ut': h_ut,
            'h_bs': h_bs,
            'max': max,
            'abs': abs
        }
        # --- [修改结束] ---

        # 循环处理每个 LSP 参数
        for p in param_order:
            # 获取均值和标准差的字符串公式
            mu_str = lsp_params.get(f'mu_log10_{p}')
            sigma_str = lsp_params.get(f'sigma_log10_{p}')

            # 检查参数是否存在
            if mu_str is None or sigma_str is None:
                raise ValueError(f"参数 '{p}' 的 mu_log10_{p} 或 sigma_log10_{p} 在 InH-Office 的 {lsp_set} 中未定义。请检查参数文件。")

            # 使用 eval() 计算均值和标准差
            mu.append(eval(mu_str, {"__builtins__": {}}, local_scope))
            sigma.append(eval(sigma_str, {"__builtins__": {}}, local_scope))

        if use_median_lsps:
            # 如果使用中值，则直接使用均值（对数值）
            x = np.array(mu)
        else:
            # 生成相关的对数正态随机变量
            # 1. 构建相关矩阵 C
            C = np.identity(len(param_order))
            for i, p1 in enumerate(param_order):
                for j, p2 in enumerate(param_order):
                    if i < j:
                        key = f'{p1}_vs_{p2}'
                        if key in correlations:
                            corr_value = correlations.get(key)
                            if corr_value is not None:
                                C[i, j] = C[j, i] = corr_value

            # 2. Cholesky 分解 C = L * L^T，处理非正定情况
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues[eigenvalues < 1e-12] = 1e-12
                C_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                L = np.linalg.cholesky(C_new)

            # 3. 生成 N 个独立的标准正态分布随机变量 z
            z = np.random.randn(len(param_order))
            # 4. 生成相关的正态分布随机变量 y = L * z
            y = L @ z
            # 5. 应用均值和标准差 x = mu + y * sigma (得到对数值)
            x = np.array(mu) + y * np.array(sigma)

        # 将对数值结果存入字典
        lsp_dict = dict(zip(param_order, x))

        # --- [已修改：使用 lsp_dict['ZSD']] ---
        # 创建并返回 LargeScaleParameters 对象，将对数值转换为线性值
        return LargeScaleParameters(
            delay_spread_s=10**lsp_dict['DS'],              # DS (秒)
            departure_azimuth_spread_deg=10**lsp_dict['ASD'], # ASD (度)
            arrival_azimuth_spread_deg=10**lsp_dict['ASA'],   # ASA (度)
            arrival_zenith_spread_deg=10**lsp_dict['ZSA'],    # ZSA (度)
            departure_zenith_spread_deg=10**lsp_dict['ZSD'],  # ZSD (度) <--- 修正
            shadow_fade_db=lsp_dict['SF'],                   # SF (dB)
            # K 因子 (线性)，如果不存在 (NLOS) 则为 0
            k_factor_linear=10**(lsp_dict['K']/10) if 'K' in lsp_dict else 0.0
        )
        # --- [修改结束] ---

    # --- [新方法：添加 get_zod_offset] ---
    def get_zod_offset(self, config: ScenarioConfig, is_los: bool) -> float:
        """
        为 InH-Office 场景实现 ZOD（离开天顶角）平均偏移量的计算。
        参考 TR 38.901 V19.0.0 的 Table 7.5-11。
        """
        # 根据 Table 7.5-11，InH-Office LOS 和 NLOS 条件下，偏移量均为 0
        return 0.0
    # --- [添加结束] ---