"""F3 单测 — geometric_native_fit 聚类 (P2-14)。"""
import numpy as np
import pytest

from mimo_ota_simulator.geometric_native_fit import (
    F64CapabilityProfile,
    MPC,
    geometric_native_fit,
)

V = [40.0, 0.0, 0.0]
FC = 3.5e9


def _grp(aoa, n, delay, jitter=2.0, power=1.0):
    """一组 n 条小角扩展 MPC (同 AoA 附近, 同时延)。"""
    offs = np.linspace(-jitter, jitter, n)
    return [MPC(delay_s=delay, power_linear=power, aoa_deg=aoa + o, aod_deg=-aoa) for o in offs]


# ───────────── 基本聚类 ─────────────

def test_two_delays_two_clusters():
    mpcs = _grp(30, 6, delay=0.0) + _grp(80, 6, delay=200e-9)
    prof = F64CapabilityProfile(name='t', tap_budget=24, rho_thresh=0.06)
    profile = geometric_native_fit(mpcs, V, FC, prof)
    assert len(profile.snapshots[0].clusters) == 2
    assert profile.clustering_meta.algorithm == 'geometric_native_fit'
    assert profile.clustering_meta.target_path == 'B2_parametric'


def test_output_is_valid_b2_profile():
    """所有簇 doppler_repr ∈ {native, dual_gaussian} (否则 §5.4 构造就 raise)。"""
    mpcs = _grp(45, 8, delay=0.0) + _grp(20, 8, delay=100e-9) + _grp(70, 8, delay=300e-9)
    profile = geometric_native_fit(mpcs, V, FC, F64CapabilityProfile(rho_thresh=0.06))
    for c in profile.snapshots[0].clusters:
        assert c.doppler_repr.kind in ('native', 'dual_gaussian')
    assert profile.clustering_meta.max_native_fit_residual is not None


# ───────────── tap 预算 ─────────────

def test_tap_budget_enforced():
    """10 个不同时延 → 初始 10 簇, budget=3 → 合并到 ≤3。"""
    mpcs = []
    for i in range(10):
        mpcs += _grp(20 + i * 7, 3, delay=i * 100e-9)
    profile = geometric_native_fit(mpcs, V, FC, F64CapabilityProfile(tap_budget=3, rho_thresh=0.06))
    assert len(profile.snapshots[0].clusters) <= 3
    assert profile.clustering_meta.tap_budget_used <= 3


# ───────────── 分裂 ─────────────

def test_multimodal_cluster_splits_with_parent_lineage():
    """同时延双峰跨速度轴 (5°/95° → Doppler +465/-40) + 收紧 rho_thresh → 分裂, 子簇带 parent。"""
    mpcs = _grp(5, 6, delay=0.0) + _grp(95, 6, delay=0.0)
    profile = geometric_native_fit(
        mpcs, V, FC, F64CapabilityProfile(tap_budget=24, rho_thresh=0.03))
    clusters = profile.snapshots[0].clusters
    assert len(clusters) >= 2                          # 分裂发生
    assert any(c.parent_cluster_id is not None for c in clusters)   # 有血缘


def test_split_respects_budget():
    """同样多峰但 budget=1 → 不能分, 只 1 簇 (best-effort native)。"""
    mpcs = _grp(5, 6, delay=0.0) + _grp(95, 6, delay=0.0)
    profile = geometric_native_fit(
        mpcs, V, FC, F64CapabilityProfile(tap_budget=1, rho_thresh=0.03))
    assert len(profile.snapshots[0].clusters) == 1


# ───────────── 确定性 / 边缘 ─────────────

def test_deterministic():
    mpcs = _grp(33, 6, delay=0.0) + _grp(77, 6, delay=150e-9)
    prof = F64CapabilityProfile(rho_thresh=0.05)
    p1 = geometric_native_fit(mpcs, V, FC, prof)
    p2 = geometric_native_fit(mpcs, V, FC, prof)
    assert p1.model_dump() == p2.model_dump()


def test_empty_mpcs_raises():
    with pytest.raises(ValueError, match='不能为空'):
        geometric_native_fit([], V, FC, F64CapabilityProfile())


def test_subrays_carried_through():
    """每簇携带其 MPC 子径 (供下游探头 PAS 映射)。"""
    mpcs = _grp(30, 5, delay=0.0)
    profile = geometric_native_fit(mpcs, V, FC, F64CapabilityProfile())
    c = profile.snapshots[0].clusters[0]
    assert c.subrays is not None and len(c.subrays) == 5
