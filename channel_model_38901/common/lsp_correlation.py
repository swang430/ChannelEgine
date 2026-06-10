"""LSP 互相关矩阵构建 (TR 38.901 §7.5 step 4 / Table 7.5-6)。

历史 bug (audit batch 2, 发现 A1): 7 个场景 model 各自复制的构建代码只按
``f'{p1}_vs_{p2}'`` 单方向查 key, 而参数 JSON 存的是反方向 (如 ``ASA_vs_DS``),
导致 11 个相关条目中 7 个被静默丢弃、Cholesky 用的相关矩阵接近单位阵。
本模块统一做双向查找, 并对无法识别的 JSON key 发 warning, 避免 dead config
再次无声出现。
"""

import warnings

import numpy as np

# Table 7.5-6 定义的全部 LSP 名 (含 ZSD, 虽然当前 ZSD 在 6 维相关采样之外)。
# 用于区分 "参数被合法移除 (如 K='N/A')" 与 "key 拼写错误 / dead config"。
KNOWN_LSP_NAMES = frozenset({'DS', 'ASD', 'ASA', 'ZSA', 'ZSD', 'SF', 'K'})


def build_lsp_correlation_matrix(param_order: list, correlations: dict) -> np.ndarray:
    """按 param_order 顺序构建对称的 LSP 互相关矩阵。

    Args:
        param_order: 本次采样使用的 LSP 名顺序 (如 ['DS','ASD','ASA','ZSA','SF','K'],
            K 为 'N/A' 的场景会事先移除 'K')。
        correlations: 参数 JSON 的 correlations dict, key 形如 'ASA_vs_DS'。
            两个方向 ('A_vs_B' / 'B_vs_A') 等价。

    Returns:
        (n, n) 对称矩阵, 对角线为 1。

    Raises:
        ValueError: 同一参数对的两个方向 key 同时存在且数值不一致
            (参数文件自相矛盾, 不允许静默二选一)。
    """
    index = {p: i for i, p in enumerate(param_order)}
    C = np.identity(len(param_order))
    seen = {}
    for key, value in correlations.items():
        p1, sep, p2 = key.partition('_vs_')
        if not sep or p1 not in KNOWN_LSP_NAMES or p2 not in KNOWN_LSP_NAMES:
            warnings.warn(
                f"LSP correlations key '{key}' 无法识别, 已忽略 (dead config?)",
                UserWarning,
            )
            continue
        if p1 == p2:
            warnings.warn(
                f"LSP correlations key '{key}' 是自相关, 已忽略", UserWarning,
            )
            continue
        if p1 not in index or p2 not in index:
            # 参数不在本次采样集合内 (例如 K='N/A' 被移除) — 合法跳过
            continue
        pair = frozenset((p1, p2))
        if pair in seen and not np.isclose(seen[pair], value):
            raise ValueError(
                f"correlations 同时给出 '{p1}_vs_{p2}' 与 '{p2}_vs_{p1}' 且数值不一致: "
                f"{value} vs {seen[pair]}"
            )
        seen[pair] = value
        i, j = index[p1], index[p2]
        C[i, j] = C[j, i] = value
    return C
