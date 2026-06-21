"""PR-1a — simulator.synthesize_ota 的 cluster_subrays 旁路 (P2-14 B-1 标注式烘焙)。

验证: 给定簇用真实子径 (确定性 phase_rad 相干求和) 替代标准扇 20 子径展开,
未给定簇走标准扇旧路径 (bit-identical)。逐位 bit-identical 回归 (改造前后 ray 路径)
已在开发期用 main 对比脚本确认 (array_equal=True); 这里固化机制语义。
"""
import numpy as np
import pytest

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import (
    ChamberConfig, TargetChannelConfig, AntennaArrayConfig, UniversalChannelProfile,
)
from mimo_ota_simulator.annotated_cdl_schema import SubRay
from channel_model_38901.common.data_models import EngineCluster


def _cluster(power=1.0, aoa=20.0, aod=10.0):
    return EngineCluster(
        delay=0.0, power=power, aod_phi=aod, aoa_phi=aoa,
        zod_theta=90.0, zoa_theta=90.0,
        as_aod=2.0, as_aoa=5.0, as_zod=1.0, as_zoa=1.0,
        k_n=1.0, initial_phases=None, xpr_linear=None,
    )


def _setup(clusters=None):
    chamber = ChamberConfig(num_probes=8, dual_polarized=False,
                            distribution='ring', radius_m=1.0)
    channel = TargetChannelConfig(
        model_name='UMi-StreetCanyon', cluster_model_name='CDL-A',
        center_frequency_hz=3.5e9,
        tx_antenna=AntennaArrayConfig(num_cols=2, array_type='ULA'),
        rx_antenna=AntennaArrayConfig(num_cols=2, array_type='ULA'),
    )
    profile = UniversalChannelProfile(
        center_frequency_hz=3.5e9, ue_velocity=np.array([10.0, 0.0, 0.0]),
        pathloss_db=80.0, clusters=clusters or [_cluster()], is_los=False,
    )
    return chamber, channel, profile


def _rays_of(res, cluster_index):
    return [r for r in res['ray_components'] if r['cluster_index'] == cluster_index]


def _sub(aoa, aod, power, phase, zoa=90.0, zod=90.0):
    return SubRay(aoa_deg=aoa, aod_deg=aod, zoa_deg=zoa, zod_deg=zod,
                  power_linear=power, phase_rad=phase)


# ───────────── 标准扇旧路径不变 ─────────────

def test_standard_fan_no_subrays_keeps_20_rays():
    """cluster_subrays=None → 每簇仍展开标准扇 20 子径。"""
    chamber, channel, profile = _setup()
    np.random.seed(0)
    res = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray')
    assert len(_rays_of(res, 0)) == 20


# ───────────── 真实子径替代 ─────────────

def test_subray_count_matches_input():
    """给 N 个真实子径 → 该簇 ray_components 数 = N (非 20)。"""
    chamber, channel, profile = _setup()
    subrays = {0: [_sub(20, 10, 1.0, 0.5), _sub(25, 12, 0.5, 1.2),
                   _sub(18, 9, 0.3, -0.8)]}
    res = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray', cluster_subrays=subrays)
    assert len(_rays_of(res, 0)) == 3


def test_phase_rad_is_used_as_phi_init():
    """子径 phase_rad 作确定性 phi_init (scalar), 非随机。"""
    chamber, channel, profile = _setup()
    subrays = {0: [_sub(20, 10, 1.0, 0.777)]}
    np.random.seed(123)
    res = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray', cluster_subrays=subrays)
    rc = _rays_of(res, 0)
    assert np.ndim(rc[0]['phi_init']) == 0           # scalar, 不是 per-probe 向量
    assert np.isclose(float(rc[0]['phi_init']), 0.777)


def test_subray_sum_deterministic_across_seeds():
    """真实子径确定性: 不同随机 seed 产出同一结果 (相干求和无随机相位)。"""
    chamber, channel, profile = _setup()
    subrays = {0: [_sub(20, 10, 1.0, 0.5), _sub(25, 12, 0.5, 1.2)]}
    np.random.seed(1)
    r1 = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray', cluster_subrays=subrays)
    np.random.seed(98765)
    r2 = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray', cluster_subrays=subrays)
    assert np.array_equal(r1['transmit_mapping_matrix_A'],
                          r2['transmit_mapping_matrix_A'])


def test_subray_power_is_relative_not_absolute():
    """子径功率按簇内相对值分配簇功率: 单子径无论 power_linear 多少都占 100%
    → p_ray = cluster.power, W_base 与 power_linear 绝对值无关。"""
    chamber, channel, profile = _setup()  # cluster.power = 1.0
    ra = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray',
        cluster_subrays={0: [_sub(20, 10, 1.0, 0.3)]})
    rb = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray',
        cluster_subrays={0: [_sub(20, 10, 99.0, 0.3)]})
    assert np.allclose(_rays_of(ra, 0)[0]['W_base'], _rays_of(rb, 0)[0]['W_base'])


def test_subray_power_split_two_rays_sums_to_cluster_power():
    """两等功率子径各分得簇功率一半: |W_base|^2 比例 ≈ 0.5/0.5。"""
    chamber, channel, profile = _setup()  # power=1.0
    subrays = {0: [_sub(20, 10, 1.0, 0.0), _sub(20, 10, 1.0, 0.0)]}  # 同角同相, 等功率
    res = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray', cluster_subrays=subrays)
    rays = _rays_of(res, 0)
    p0 = float(np.sum(np.abs(rays[0]['W_base'])**2))
    p1 = float(np.sum(np.abs(rays[1]['W_base'])**2))
    assert np.isclose(p0, p1)                          # 等功率
    # 单子径 (占 100%) 的功率应 ≈ 两子径之和 (各 50%)
    res1 = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray',
        cluster_subrays={0: [_sub(20, 10, 1.0, 0.0)]})
    p_full = float(np.sum(np.abs(_rays_of(res1, 0)[0]['W_base'])**2))
    assert np.isclose(p0 + p1, p_full, rtol=1e-9)


# ───────────── 旁路隔离 ─────────────

def test_bypass_only_affects_named_clusters():
    """只给 cluster 0 真实子径; cluster 1 仍走标准扇 20。"""
    chamber, channel, profile = _setup(clusters=[
        _cluster(power=0.6, aoa=20, aod=10),
        _cluster(power=0.4, aoa=120, aod=80),
    ])
    subrays = {0: [_sub(20, 10, 1.0, 0.5), _sub(22, 11, 0.5, 1.0)]}
    np.random.seed(0)
    res = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray', cluster_subrays=subrays)
    assert len(_rays_of(res, 0)) == 2     # 真实子径
    assert len(_rays_of(res, 1)) == 20    # 标准扇旧路径


def test_unannotated_cluster_rng_isolated():
    """Codex P2: 给 cluster 0 加 (或改变子径数) 不影响未注解 cluster 1 (标准扇) 的
    随机实现 —— 每簇独立派生 RNG, 簇间隔离 (确定性簇跳过随机抽取不再偏移共享流)。"""
    chamber, channel, profile = _setup(clusters=[
        _cluster(power=0.6, aoa=20, aod=10),
        _cluster(power=0.4, aoa=120, aod=80),
    ])
    np.random.seed(42)
    r1 = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray',
        cluster_subrays={0: [_sub(20, 10, 1.0, 0.5)]})            # cluster 0: 1 子径
    c1_a = [r['phi_init'] for r in _rays_of(r1, 1)]
    np.random.seed(42)
    r2 = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray',
        cluster_subrays={0: [_sub(20, 10, 1, 0.5), _sub(21, 11, 1, 0.6),
                             _sub(22, 12, 1, 0.7)]})              # cluster 0: 3 子径
    c1_b = [r['phi_init'] for r in _rays_of(r2, 1)]
    # cluster 1 标准扇随机实现不受 cluster 0 子径数影响 (Codex P2 修复前会偏移)
    assert np.array_equal(c1_a, c1_b)
    assert len(c1_a) == 20
