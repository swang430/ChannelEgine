"""F1 — 簇几何 → F64 原生多普勒谱参数映射 (P2-14 B-2 信道注入).

把一个簇的角度结构 (子径到达方向 + 功率) + UE 速度矢量 + 载频, 映射成
PROPSIM F64 参数化衰落引擎能直接生成的【原生谱】参数 (Classical / Gaussian /
双高斯), 并给出拟合残差 —— 供 geometric_native_fit 聚类 (F3) 判定每簇是否
native 可表示 (残差超阈则触发分裂)。

设计依据: MIMO-First `docs/design/RT-MPDB-CDL-F64-channel-injection-design_V1.0.md`
§5.3 / §7。F64 仅有闭式谱集、无任意 PSD 参数口 (手册证实), 故 RT 簇的定制多普勒
展宽需用 native 谱 (质心+展宽) 或双高斯逼近。

多普勒约定与 `simulator.py` 一致 (line ~931-943):
    k_hat = [sinθcosφ, sinθsinφ, cosθ]   (θ=ZoA from zenith[deg], φ=AoA[deg])
    f_D   = (v · k_hat) / λ,   λ = c / f_c,  c = 3e8

纯函数, 仅依赖 numpy。无硬件 / 无 I/O / 无随机性 (双高斯 EM 用确定性 percentile 初始化)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

C_LIGHT = 3e8

# TR 38.901 Table 7.5-3 标准 20 子径角度偏置 (与 simulator.py default_ray_offsets 一致)
DEFAULT_RAY_OFFSETS = np.array([
    0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715,
    0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481,
    1.5195, -1.5195, 2.1551, -2.1551,
])

NativeShape = Literal['classical', 'gaussian', 'dual_gaussian', 'flat', 'pure_doppler']


@dataclass(frozen=True)
class NativeMappingResult:
    """一个簇映射到 F64 原生谱的结果。

    `kind` 对齐 V1.0 §5.3 的 doppler_repr 区分联合体:
      - 'native'        → shape ∈ {classical, gaussian, flat, pure_doppler}
      - 'dual_gaussian' → 双高斯逼近
    """
    kind: Literal['native', 'dual_gaussian']
    shape: NativeShape
    f_d_centroid_hz: float        # 质心 = Σ p·f_D / Σ p
    f_d_max_hz: float             # 展宽半宽 = max|f_D − 质心| (绕质心; 质心见 f_d_centroid_hz)
    params: dict                  # gaussian:{std_hz}; dual_gaussian:{beam_a,beam_b,power_ratio_db}; ...
    residual: float               # 总变差距离 TV ∈ [0,1], 越小越好
    needs_split: bool             # 即便双高斯残差仍超阈 → F3 应分裂该簇


# ───────────────────────── 子径多普勒 ─────────────────────────

def _doppler_from_angles(aoa_deg, zoa_deg, velocity_mps, f_c_hz) -> np.ndarray:
    """f_D = (v · k_hat)/λ。与 simulator.py 约定一致。返回与角度同形状的数组。"""
    v = np.asarray(velocity_mps, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"velocity_mps 必须是 3 维矢量 (vx,vy,vz), got shape {v.shape}")
    if f_c_hz <= 0:
        raise ValueError(f"f_c_hz 必须 > 0, got {f_c_hz}")
    lam = C_LIGHT / f_c_hz
    th = np.deg2rad(np.asarray(zoa_deg, dtype=float))   # ZoA from zenith
    ph = np.deg2rad(np.asarray(aoa_deg, dtype=float))   # AoA
    k_hat = np.stack([np.sin(th) * np.cos(ph),
                      np.sin(th) * np.sin(ph),
                      np.cos(th)], axis=-1)              # (..., 3)
    return (k_hat @ v) / lam


def subray_dopplers_from_geometry(
    aoa_deg: float, zoa_deg: float,
    as_aoa_deg: float, as_zoa_deg: float,
    velocity_mps: Sequence[float], f_c_hz: float,
    ray_offsets: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从簇均值角 + 角扩展 (标准 38.901 扇) 生成子径多普勒 + 等功率。

    返回 (f_D[N], power[N])，power 等分 (Σ=1，N=len(ray_offsets)，默认 20)。
    """
    offs = DEFAULT_RAY_OFFSETS if ray_offsets is None else np.asarray(ray_offsets, dtype=float)
    aoa = aoa_deg + offs * as_aoa_deg
    zoa = zoa_deg + offs * as_zoa_deg
    fd = _doppler_from_angles(aoa, zoa, velocity_mps, f_c_hz)
    power = np.full(fd.shape, 1.0 / fd.size)
    return fd, power


def subray_dopplers_from_rays(
    subray_aoa_deg: Sequence[float], subray_zoa_deg: Sequence[float],
    subray_power: Sequence[float],
    velocity_mps: Sequence[float], f_c_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """从显式 RT 子径 (任意数量/角度/功率) 生成多普勒 + 归一化功率 (Σ=1)。"""
    aoa = np.asarray(subray_aoa_deg, dtype=float)
    zoa = np.asarray(subray_zoa_deg, dtype=float)
    p = np.asarray(subray_power, dtype=float)
    if aoa.shape != zoa.shape or aoa.shape != p.shape:
        raise ValueError("subray aoa/zoa/power 形状必须一致")
    if p.size == 0:
        raise ValueError("子径不能为空")
    total = p.sum()
    if total <= 0:
        raise ValueError("subray_power 之和必须 > 0")
    fd = _doppler_from_angles(aoa, zoa, velocity_mps, f_c_hz)
    return fd, p / total


# ───────────────────────── 谱密度 ─────────────────────────

def _gaussian_density(f: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-9)
    return np.exp(-0.5 * ((f - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _jakes_density(f: np.ndarray, f_d_max: float) -> np.ndarray:
    """Clarke/Jakes U 型谱 (各向同性)。|f|<f_d_max 有定义，边界 ∝ 1/√(1-(f/fmax)²)。"""
    f_d_max = max(float(f_d_max), 1e-9)
    out = np.zeros_like(f, dtype=float)
    inside = np.abs(f) < f_d_max * (1.0 - 1e-6)
    out[inside] = 1.0 / (np.pi * f_d_max * np.sqrt(1.0 - (f[inside] / f_d_max) ** 2))
    return out


# ───────────────────────── 矩 + 拟合 ─────────────────────────

def psd_moments(fd: np.ndarray, power: np.ndarray) -> dict:
    """加权矩。

    - ``centroid_hz``     : 质心 Σp·f
    - ``std_hz``          : 加权标准差
    - ``f_d_max_hz``      : 展宽半宽 = max|f − 质心| (绕质心, 与 V1.0 §5.3 结果字段一致;
                            质心由 f_d_centroid 单独承载, 不混进展宽)
    - ``f_d_abs_max_hz``  : 绝对最大 |f| (用于 flat 判定 —— "有没有任何多普勒")
    """
    w = power / power.sum()
    centroid = float(np.sum(w * fd))
    var = float(np.sum(w * (fd - centroid) ** 2))
    if fd.size == 0:
        return {'centroid_hz': centroid, 'std_hz': 0.0, 'f_d_max_hz': 0.0, 'f_d_abs_max_hz': 0.0}
    return {
        'centroid_hz': centroid,
        'std_hz': float(np.sqrt(max(var, 0.0))),
        'f_d_max_hz': float(np.max(np.abs(fd - centroid))),
        'f_d_abs_max_hz': float(np.max(np.abs(fd))),
    }


def fit_single_gaussian(fd: np.ndarray, power: np.ndarray) -> tuple[float, float]:
    """矩匹配单高斯 → (mu, sigma)。"""
    m = psd_moments(fd, power)
    return m['centroid_hz'], m['std_hz']


def fit_dual_gaussian(fd: np.ndarray, power: np.ndarray, n_iter: int = 60) -> dict:
    """1D 加权两分量 GMM (确定性 EM, percentile 初始化)。

    返回 {beam_a:{center_shift_hz,std_hz}, beam_b:{...}, power_ratio_db, weight_a}。
    power_ratio_db = 10·log10(w_a / w_b)。
    """
    w_obs = power / power.sum()
    # 确定性初始化: 两分量中心放在 25/75 分位 (按 Doppler 排序的加权分位)
    order = np.argsort(fd)
    fd_s, w_s = fd[order], w_obs[order]
    cdf = np.cumsum(w_s)
    mu_a = float(fd_s[np.searchsorted(cdf, 0.25)])
    mu_b = float(fd_s[min(np.searchsorted(cdf, 0.75), fd_s.size - 1)])
    spread = psd_moments(fd, power)['std_hz']
    sig_a = sig_b = max(spread * 0.5, 1e-3)
    wa, wb = 0.5, 0.5
    if mu_a == mu_b:                 # 退化: 全部同频 → 拉开一点避免除零
        mu_b = mu_a + max(spread, 1.0)

    for _ in range(n_iter):
        da = wa * _gaussian_density(fd, mu_a, sig_a)
        db = wb * _gaussian_density(fd, mu_b, sig_b)
        denom = da + db
        denom = np.where(denom <= 0, 1e-300, denom)
        ra = da / denom              # 责任度 a
        rb = 1.0 - ra
        # M-step (按观测权重加权)
        na = float(np.sum(ra * w_obs))
        nb = float(np.sum(rb * w_obs))
        if na <= 1e-12 or nb <= 1e-12:
            break
        mu_a = float(np.sum(ra * w_obs * fd) / na)
        mu_b = float(np.sum(rb * w_obs * fd) / nb)
        sig_a = float(np.sqrt(max(np.sum(ra * w_obs * (fd - mu_a) ** 2) / na, 1e-6)))
        sig_b = float(np.sqrt(max(np.sum(rb * w_obs * (fd - mu_b) ** 2) / nb, 1e-6)))
        wa, wb = na, nb

    # 规范化: 让 beam_a 总是低频侧, 保证结果稳定可测
    if mu_a > mu_b:
        mu_a, mu_b = mu_b, mu_a
        sig_a, sig_b = sig_b, sig_a
        wa, wb = wb, wa
    power_ratio_db = float(10.0 * np.log10(max(wa, 1e-12) / max(wb, 1e-12)))
    return {
        'beam_a': {'center_shift_hz': mu_a, 'std_hz': sig_a},
        'beam_b': {'center_shift_hz': mu_b, 'std_hz': sig_b},
        'power_ratio_db': power_ratio_db,
        'weight_a': float(wa / (wa + wb)),
    }


# ───────────────────────── 残差 (CDF-L1 距离) ─────────────────────────
# 用累积分布 (CDF) 的平均绝对差, 而非 PDF 的总变差 —— CDF 对"20 个离散射线 vs
# 连续谱"这种稀疏点集稳健 (PDF-TV 会因离散性 / bin 错位虚高), 且仍 ∈ [0,1]。

def _grid_and_emp_cdf(fd: np.ndarray, power: np.ndarray, n_bins: int = 256):
    """覆盖子径的细网格上的加权经验 CDF → (centers, width, emp_cdf[递增到 1])。"""
    m = psd_moments(fd, power)
    half = max(m['f_d_max_hz'], m['std_hz'] * 4.0, 1.0)
    lo = m['centroid_hz'] - half - m['std_hz']
    hi = m['centroid_hz'] + half + m['std_hz']
    if hi - lo < 1e-9:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    hist, _ = np.histogram(fd, bins=edges, weights=power)
    s = hist.sum()
    emp = hist / s if s > 0 else hist
    return centers, width, np.cumsum(emp)


def _cdf_l1(emp_cdf: np.ndarray, cand_density_at_centers: np.ndarray, width: float) -> float:
    """候选谱在网格上归一化 → 累积 → 与经验 CDF 的平均绝对差 ∈ [0,1]。"""
    q = cand_density_at_centers * width
    qs = q.sum()
    q = q / qs if qs > 0 else q
    cand_cdf = np.cumsum(q)
    return float(np.mean(np.abs(emp_cdf - cand_cdf)))


# ───────────────────────── 顶层映射 ─────────────────────────

def map_doppler_to_native(
    fd: np.ndarray, power: np.ndarray,
    *,
    residual_margin: float = 0.01,
    split_threshold: float = 0.06,
    pure_doppler_std_hz: float = 1.0,
    flat_f_d_max_hz: float = 1e-3,
) -> NativeMappingResult:
    """核心: 把子径多普勒分布映射到 F64 原生谱 (含残差 + needs_split)。

    选择策略: 在 {classical, gaussian, dual_gaussian} 中取残差最小者; 残差接近
    (≤ best+margin) 时偏好更简单的谱形 (classical < gaussian < dual_gaussian),
    避免过拟合。即便最优残差仍 > split_threshold → needs_split=True (交 F3 分裂)。
    """
    fd = np.asarray(fd, dtype=float)
    power = np.asarray(power, dtype=float)
    if fd.size == 0 or power.sum() <= 0:
        raise ValueError("fd 为空或 power 之和 ≤ 0")

    m = psd_moments(fd, power)
    centroid, std, f_d_max = m['centroid_hz'], m['std_hz'], m['f_d_max_hz']
    f_d_abs_max = m['f_d_abs_max_hz']

    # 退化 1: 几乎无【任何】多普勒 (静止) → flat。用绝对 max 判, 否则斜入射单线
    # (展宽≈0 但质心≠0) 会被误判 flat。
    if f_d_abs_max <= flat_f_d_max_hz:
        return NativeMappingResult('native', 'flat', centroid, f_d_max, {}, 0.0, False)
    # 退化 2: 展宽极窄 (单射线 / LOS) → pure_doppler (质心处单线)
    if std <= pure_doppler_std_hz:
        return NativeMappingResult('native', 'pure_doppler', centroid, f_d_max,
                                   {}, 0.0, False)

    centers, width, emp_cdf = _grid_and_emp_cdf(fd, power)

    # 候选 1: classical (Jakes)。Jakes 居中到【质心】再评分 (质心由 f_d_centroid 单独承载),
    # 否则斜入射簇 (质心≠0) 跟零中心 Jakes 比会虚高残差 (Codex P2 #34)。半宽 = 绕质心展宽。
    rho_classical = _cdf_l1(emp_cdf, _jakes_density(centers - centroid, f_d_max), width)
    # 候选 2: 单高斯
    g_mu, g_sig = fit_single_gaussian(fd, power)
    rho_gauss = _cdf_l1(emp_cdf, _gaussian_density(centers, g_mu, g_sig), width)
    # 候选 3: 双高斯
    dg = fit_dual_gaussian(fd, power)
    dg_density = (dg['weight_a'] * _gaussian_density(centers, dg['beam_a']['center_shift_hz'], dg['beam_a']['std_hz'])
                  + (1 - dg['weight_a']) * _gaussian_density(centers, dg['beam_b']['center_shift_hz'], dg['beam_b']['std_hz']))
    rho_dual = _cdf_l1(emp_cdf, dg_density, width)

    best = min(rho_classical, rho_gauss, rho_dual)
    # needs_split 基于【最优可达 native 残差 best】: 即便最贴合的原生谱(含双高斯)
    # 都拟合不动, 才交 F3 分裂; 跟所选谱(可能因 margin 偏好更简单者)无关。
    needs_split = best > split_threshold
    # 简单优先: classical → gaussian → dual_gaussian
    if rho_classical <= best + residual_margin:
        return NativeMappingResult('native', 'classical', centroid, f_d_max,
                                   {}, rho_classical, needs_split)
    if rho_gauss <= best + residual_margin:
        return NativeMappingResult('native', 'gaussian', centroid, f_d_max,
                                   {'std_hz': g_sig}, rho_gauss, needs_split)
    return NativeMappingResult('dual_gaussian', 'dual_gaussian', centroid, f_d_max,
                               {'beam_a': dg['beam_a'], 'beam_b': dg['beam_b'],
                                'power_ratio_db': dg['power_ratio_db']},
                               rho_dual, needs_split)


def map_cluster_to_native_from_geometry(
    aoa_deg: float, zoa_deg: float, as_aoa_deg: float, as_zoa_deg: float,
    velocity_mps: Sequence[float], f_c_hz: float,
    *, ray_offsets: Optional[Sequence[float]] = None, **kwargs,
) -> NativeMappingResult:
    """便捷入口: 簇均值角 + AS (标准扇) → 原生谱。"""
    fd, power = subray_dopplers_from_geometry(
        aoa_deg, zoa_deg, as_aoa_deg, as_zoa_deg, velocity_mps, f_c_hz, ray_offsets)
    return map_doppler_to_native(fd, power, **kwargs)


def map_cluster_to_native_from_rays(
    subray_aoa_deg: Sequence[float], subray_zoa_deg: Sequence[float],
    subray_power: Sequence[float], velocity_mps: Sequence[float], f_c_hz: float,
    **kwargs,
) -> NativeMappingResult:
    """便捷入口: 显式 RT 子径 → 原生谱 (B-2 真实路径)。"""
    fd, power = subray_dopplers_from_rays(
        subray_aoa_deg, subray_zoa_deg, subray_power, velocity_mps, f_c_hz)
    return map_doppler_to_native(fd, power, **kwargs)
