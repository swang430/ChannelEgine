"""由接收天线之前的两极化电场求各发射端口、各簇的探头复系数。"""

import numpy as np


def fit_probe_taps(probe_field, incident_field):
    """probe_field[位置,极化,馈电口]，incident_field[发射端口,簇,位置,极化]。

    两组复电场须使用相同的空间位置、极化基、幅度单位和相位参考。
    最小二乘在所有输入位置等权拟合，不对单个端口独立归一化。
    """
    basis = np.asarray(probe_field, dtype=complex)
    target = np.asarray(incident_field, dtype=complex)
    if basis.ndim != 3 or basis.shape[1] not in (2, 3) or target.ndim != 4:
        raise ValueError("需要逐位置的双极化探头响应和分发射端口、分簇的入射场")
    if target.shape[-2:] != basis.shape[:2] or 0 in target.shape or 0 in basis.shape:
        raise ValueError("入射场与探头响应的位置、极化维度不一致或为空")
    if not np.isfinite(basis).all() or not np.isfinite(target).all():
        raise ValueError("电场含非有限复数")
    matrix = basis.reshape(-1, basis.shape[-1])
    if np.any(np.all(matrix == 0, axis=0)):
        raise ValueError("有馈电口在全部位置均无场响应")
    rhs = target.reshape(-1, matrix.shape[0]).T
    solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=1e-8)
    if not np.isfinite(solution).all():
        raise ValueError("求得的探头系数超出数值范围")
    return solution.T.reshape(*target.shape[:2], basis.shape[-1])
