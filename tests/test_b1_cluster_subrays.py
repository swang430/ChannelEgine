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


def test_empty_cluster_subrays_equals_none():
    """空 dict cluster_subrays={} = 无注解簇, 与 None 同义: 标准扇 + bit-identical,
    不崩 (Codex P2: base_seed 用 truthy=None 但 RNG 分支若 is-not-None 会 None+i TypeError)。"""
    chamber, channel, profile = _setup()
    np.random.seed(7)
    r_empty = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray', cluster_subrays={})
    np.random.seed(7)
    r_none = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray', cluster_subrays=None)
    assert len(_rays_of(r_empty, 0)) == 20          # 标准扇, 不崩
    assert np.array_equal(r_empty['transmit_mapping_matrix_A'],
                          r_none['transmit_mapping_matrix_A'])   # {} ≡ None bit-identical


def test_unannotated_cluster_matches_legacy_none():
    """Codex P2 (第三轮): 注解 cluster 0 后, 未注解 cluster 1 (标准扇) 的随机实现与
    cluster_subrays=None (全标准扇) 逐位一致 —— subray_sum 簇消耗等量 draw 丢弃, 全局
    RNG 流前进量不变, legacy 兼容 + 隔离同时成立。"""
    chamber, channel, profile = _setup(clusters=[
        _cluster(power=0.6, aoa=20, aod=10),
        _cluster(power=0.4, aoa=120, aod=80),
    ])
    np.random.seed(2024)
    r_none = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray', cluster_subrays=None)
    c1_none = [r['phi_init'] for r in _rays_of(r_none, 1)]
    np.random.seed(2024)
    r_ann = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile, synthesis_method='ray',
        cluster_subrays={0: [_sub(20, 10, 1.0, 0.5), _sub(21, 11, 0.5, 1.0)]})
    c1_ann = [r['phi_init'] for r in _rays_of(r_ann, 1)]
    assert np.array_equal(c1_none, c1_ann)   # cluster 1 与全标准扇情况逐位一致 (legacy 兼容)


def test_subray_without_phase_rad_fails_fast():
    """Codex P2: 直接调 synthesize_ota 传 phase_rad=None 的子径 → fail-fast。SubRay schema
    本身允许 phase_rad=None (仅上层 AnnotatedCDLProfile 对 subray_sum 拒绝), 但底层旁路也
    必须守 —— 否则 None 静默落随机分支, 失去确定性 + 多抽 RNG 污染后续未注解簇。"""
    chamber, channel, profile = _setup()
    bad = {0: [SubRay(aoa_deg=20, aod_deg=10, zoa_deg=90, zod_deg=90,
                      power_linear=1.0, phase_rad=None)]}
    with pytest.raises(ValueError, match="phase_rad"):
        MIMO_OTA_Simulator().synthesize_ota(
            chamber, channel, profile, synthesis_method='ray', cluster_subrays=bad)


def test_empty_subray_list_for_cluster_fails_fast():
    """Codex P2: {i: []} 显式注解该簇但子径为空 → fail-fast (上游丢光子径信号),
    不静默退标准扇假射线。区别于 {} / None (整体未注解, 走标准扇 bit-identical)。"""
    chamber, channel, profile = _setup()
    with pytest.raises(ValueError, match="空列表|子径"):
        MIMO_OTA_Simulator().synthesize_ota(
            chamber, channel, profile, synthesis_method='ray', cluster_subrays={0: []})
