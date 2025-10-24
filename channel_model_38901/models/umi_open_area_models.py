import numpy as np
import math
from .abc_models import ABC_ChannelConditionModel, ABC_PathlossModel, ABC_FadingModel
from channel_model_38901.common.data_models import ScenarioConfig, LargeScaleParameters
from channel_model_38901.parameters.parameter_loader import ParameterTables

class UMi_OpenArea_ChannelCondition(ABC_ChannelConditionModel):
    """UMi-OpenArea 场景的信道条件模型实现。"""
    def get_condition(self, distance_2d: float, h_ut: float) -> bool:
        """
        计算 UMi-OpenArea 场景的 LOS 概率。
        参考 TR 38.901 表 7.4.2-1。对于 UMi-OpenArea，总是 LOS。
        """
        return True # UMi-OpenArea 总是 LOS

class UMi_OpenArea_Pathloss(ABC_PathlossModel):
    """UMi-OpenArea 场景的路径损耗模型实现。"""
    def __init__(self, params: ParameterTables):
        # 从参数加载器获取 UMi-OpenArea 场景的特定参数
        self.params = params.get_scenario_params('UMi-OpenArea')

    def calculate_pathloss(self, config: ScenarioConfig, is_los: bool) -> float:
        """
        计算 UMi-OpenArea 场景的路径损耗（dB）。
        参考 TR 38.901 表 7.4.1-1。UMi-OpenArea 只有 LOS 模型。
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
            'np': np # 允许使用 numpy 函数 (如果需要的话)
        }

        # 获取 LOS 路径损耗参数
        los_params = self.params['pathloss_los']
        # 获取公式字符串
        formula_1_str = los_params['formula_1']
        # 使用 eval() 计算路径损耗，限制可访问的内建函数
        pathloss = eval(formula_1_str, {"__builtins__": {}}, local_scope)

        return pathloss

class UMi_OpenArea_Fading(ABC_FadingModel):
    """UMi-OpenArea 场景的大尺度衰落模型实现。"""
    def __init__(self, params: ParameterTables):
        # 从参数加载器获取 UMi-OpenArea 场景的特定参数
        self.params = params.get_scenario_params('UMi-OpenArea')

    # --- [已修改：添加 config 参数] ---
    def generate_lsps(self, config: ScenarioConfig, is_los: bool, fc_ghz: float, use_median_lsps: bool = False) -> LargeScaleParameters:
        """
        为 UMi-OpenArea 场景生成大尺度参数。
        参考 TR 38.901 Section 7.5 和 Table 7.5-6。
        UMi-OpenArea 只有 LOS 参数。
        """
        lsp_set = 'lsp_parameters_los' # UMi-OpenArea 只有 LOS
        lsp_params = self.params[lsp_set]
        correlations = lsp_params['correlations']

        # --- [已修改：添加 'ZSD'] ---
        param_order = ['DS', 'ASD', 'ASA', 'ZSA', 'ZSD', 'SF', 'K']
        k_is_na = lsp_params.get(f'mu_log10_K') == 'N/A'
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

            # 检查参数是否存在 (特别是 ZSD 可能在旧版参数文件中缺失)
            if mu_str is None or sigma_str is None:
                raise ValueError(f"参数 '{p}' 的 mu_log10_{p} 或 sigma_log10_{p} 在 UMi-OpenArea 的 lsp_parameters_los 中未定义。请检查参数文件。")

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
                            # 检查相关系数是否存在，不存在则默认为 0
                            corr_value = correlations.get(key)
                            if corr_value is not None:
                                C[i, j] = C[j, i] = corr_value
                            # else: C[i,j] = C[j,i] = 0.0 # 默认不相关

            # 2. Cholesky 分解 C = L * L^T，处理非正定情况
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # 如果 C 不是正定，找到最近的正定矩阵
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues[eigenvalues < 1e-12] = 1e-12 # 替换小的/负的特征值
                C_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                # 重新进行 Cholesky 分解
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
            # K 因子 (线性)，如果不存在则为 0
            k_factor_linear=10**(lsp_dict['K']/10) if 'K' in lsp_dict else 0.0
        )
        # --- [修改结束] ---

    # --- [新方法：添加 get_zod_offset] ---
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

            # 应用 UMi-NLOS ZOD offset 的完整公式
            offset = -10 ** (-1.5 * math.log10(max(10, d_2d))) + 3.3

            return offset