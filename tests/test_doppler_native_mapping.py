"""F1 单测 — 簇几何 → F64 原生多普勒谱映射 (P2-14)。"""
import numpy as np
import pytest

from mimo_ota_simulator.doppler_native_mapping import (
    C_LIGHT,
    map_cluster_to_native_from_geometry,
    map_cluster_to_native_from_rays,
    psd_moments,
    subray_dopplers_from_geometry,
    subray_dopplers_from_rays,
    _doppler_from_angles,
    fit_dual_gaussian,
)


# ───────────── 多普勒约定 (与 simulator.py 一致) ─────────────

def test_doppler_ray_along_velocity_is_fmax():
    """沿速度方向的射线 (AoA=0, ZoA=90) → f_D = v/λ = v·f_c/c。"""
    v, f_c = 30.0, 3.5e9
    fd = _doppler_from_angles(aoa_deg=0.0, zoa_deg=90.0, velocity_mps=[v, 0, 0], f_c_hz=f_c)
    assert fd == pytest.approx(v * f_c / C_LIGHT, rel=1e-9)


def test_doppler_perpendicular_ray_is_zero():
    """垂直于速度的射线 (AoA=90, ZoA=90) → f_D ≈ 0。"""
    fd = _doppler_from_angles(aoa_deg=90.0, zoa_deg=90.0, velocity_mps=[30, 0, 0], f_c_hz=3.5e9)
    assert fd == pytest.approx(0.0, abs=1e-6)


def test_doppler_centroid_is_fmax_cos_phi():
    """窄簇质心 ≈ f_D,max·cosφ (V1.0 §native 映射公式)。"""
    v, f_c, phi = 30.0, 3.5e9, 60.0
    f_d_max = v * f_c / C_LIGHT
    fd, power = subray_dopplers_from_geometry(
        aoa_deg=phi, zoa_deg=90.0, as_aoa_deg=3.0, as_zoa_deg=0.0,
        velocity_mps=[v, 0, 0], f_c_hz=f_c)
    centroid = psd_moments(fd, power)['centroid_hz']
    assert centroid == pytest.approx(f_d_max * np.cos(np.deg2rad(phi)), rel=0.05)


# ───────────── 三种谱分类 ─────────────

def test_narrow_cluster_maps_to_gaussian():
    """中窄角扩展、斜入射 → 单峰 → gaussian。"""
    res = map_cluster_to_native_from_geometry(
        aoa_deg=60.0, zoa_deg=90.0, as_aoa_deg=8.0, as_zoa_deg=2.0,
        velocity_mps=[30, 0, 0], f_c_hz=3.5e9)
    assert res.kind == 'native'
    assert res.shape == 'gaussian'
    assert 'std_hz' in res.params
    assert not res.needs_split


def test_isotropic_cluster_maps_to_classical():
    """各向同性 (AoA 均匀铺满 360°) → Jakes U 型 → classical。"""
    aoa = np.linspace(0, 360, 120, endpoint=False)
    zoa = np.full_like(aoa, 90.0)
    power = np.ones_like(aoa)
    res = map_cluster_to_native_from_rays(aoa, zoa, power, velocity_mps=[30, 0, 0], f_c_hz=3.5e9)
    assert res.shape == 'classical'
    assert res.kind == 'native'
    assert res.residual < 0.1


def test_bimodal_cluster_maps_to_dual_gaussian():
    """两个分离角度组 (不同 |cosφ| → 两个分离多普勒质心) → dual_gaussian。"""
    spread = np.linspace(-3, 3, 10)
    aoa = np.concatenate([20.0 + spread, 75.0 + spread])  # 组A cos大→高多普勒, 组B→低
    zoa = np.full_like(aoa, 90.0)
    power = np.ones_like(aoa)
    res = map_cluster_to_native_from_rays(aoa, zoa, power, velocity_mps=[40, 0, 0], f_c_hz=3.5e9)
    assert res.kind == 'dual_gaussian'
    assert res.shape == 'dual_gaussian'
    a = res.params['beam_a']['center_shift_hz']
    b = res.params['beam_b']['center_shift_hz']
    assert a < b                      # beam_a 规范化为低频侧
    assert not res.needs_split        # 双峰双高斯能拟合


# ───────────── 退化 / 边缘 ─────────────

def test_zero_velocity_maps_to_flat():
    res = map_cluster_to_native_from_geometry(
        aoa_deg=30.0, zoa_deg=90.0, as_aoa_deg=10.0, as_zoa_deg=2.0,
        velocity_mps=[0, 0, 0], f_c_hz=3.5e9)
    assert res.shape == 'flat'
    assert res.f_d_max_hz == pytest.approx(0.0, abs=1e-3)


def test_single_ray_maps_to_pure_doppler():
    """单射线 (展宽≈0) → pure_doppler 单线。"""
    res = map_cluster_to_native_from_rays(
        [45.0], [90.0], [1.0], velocity_mps=[30, 0, 0], f_c_hz=3.5e9)
    assert res.shape == 'pure_doppler'


def test_needs_split_fires_when_best_residual_exceeds_threshold():
    """needs_split 机制: 非对称多峰 + 收紧 rho_thresh → 最优 native 残差超阈 → 交 F3 分裂。

    (CDF-L1 对多峰宽容是设计正确的 —— native 谱罩得住很多; 角度多峰的分裂属
    F3 联合域。F1 的 needs_split 只是 best 残差 vs 现场标定阈值这一机制。)
    """
    aoa = np.concatenate([np.full(8, 80.0), np.full(6, 55.0), np.full(8, 20.0)])
    zoa = np.full_like(aoa, 90.0)
    power = np.ones_like(aoa)
    res = map_cluster_to_native_from_rays(aoa, zoa, power, velocity_mps=[40, 0, 0],
                                          f_c_hz=3.5e9, split_threshold=0.02)
    assert res.residual > 0.02
    assert res.needs_split


def test_clean_fit_does_not_split():
    """干净单峰拟合 (残差 ≪ 默认阈值) → needs_split=False。"""
    res = map_cluster_to_native_from_geometry(
        aoa_deg=60.0, zoa_deg=90.0, as_aoa_deg=8.0, as_zoa_deg=2.0,
        velocity_mps=[30, 0, 0], f_c_hz=3.5e9)
    assert res.residual < 0.02
    assert not res.needs_split


# ───────────── 残差界 / 确定性 ─────────────

def test_residual_bounded_0_1():
    res = map_cluster_to_native_from_geometry(
        aoa_deg=33.0, zoa_deg=85.0, as_aoa_deg=12.0, as_zoa_deg=4.0,
        velocity_mps=[25, 5, 0], f_c_hz=2.6e9)
    assert 0.0 <= res.residual <= 1.0


def test_deterministic_same_input_same_output():
    """EM 用确定性 percentile 初始化 → 两次结果完全一致。"""
    args = dict(aoa_deg=50.0, zoa_deg=88.0, as_aoa_deg=9.0, as_zoa_deg=3.0,
                velocity_mps=[30, 10, 0], f_c_hz=3.5e9)
    r1 = map_cluster_to_native_from_geometry(**args)
    r2 = map_cluster_to_native_from_geometry(**args)
    assert r1 == r2


def test_dual_gaussian_beam_a_is_low_side():
    """fit_dual_gaussian 规范化: beam_a.center ≤ beam_b.center。"""
    fd = np.array([-100.0] * 10 + [80.0] * 10)
    power = np.ones(20)
    dg = fit_dual_gaussian(fd, power)
    assert dg['beam_a']['center_shift_hz'] <= dg['beam_b']['center_shift_hz']


# ───────────── 入参校验 ─────────────

def test_bad_velocity_shape_raises():
    with pytest.raises(ValueError):
        _doppler_from_angles(0.0, 90.0, [1, 2], 3.5e9)        # 2 维而非 3 维


def test_nonpositive_fc_raises():
    with pytest.raises(ValueError):
        _doppler_from_angles(0.0, 90.0, [30, 0, 0], 0.0)


def test_empty_rays_raises():
    with pytest.raises(ValueError):
        subray_dopplers_from_rays([], [], [], [30, 0, 0], 3.5e9)


def test_zero_power_sum_raises():
    with pytest.raises(ValueError):
        subray_dopplers_from_rays([10, 20], [90, 90], [0, 0], [30, 0, 0], 3.5e9)
