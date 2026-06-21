"""F5 单测 — phase_continuous 聚类 (确定性相位, B-1/GCM 路) (P2-14)。"""
import numpy as np
import pytest

from mimo_ota_simulator.geometric_native_fit import F64CapabilityProfile, MPC
from mimo_ota_simulator.phase_continuous import phase_continuous_fit

V = [40.0, 0.0, 0.0]
FC = 3.5e9
PROF = F64CapabilityProfile(tap_budget=24)


def _grp(aoa, n, delay, phase0=0.1):
    return [MPC(delay_s=delay, power_linear=1.0, aoa_deg=aoa + o, aod_deg=-aoa,
                phase_rad=phase0 + i * 0.3)
            for i, o in enumerate(np.linspace(-2, 2, n))]


# ───────────── 基本 ─────────────

def test_produces_subray_sum_clusters_with_phase():
    mpcs = _grp(30, 5, 0.0) + _grp(80, 5, 200e-9)
    p = phase_continuous_fit(mpcs, V, FC, PROF)
    assert p.clustering_meta.algorithm == 'phase_continuous'
    assert p.clustering_meta.target_path == 'B1_baked'
    for c in p.snapshots[0].clusters:
        assert c.doppler_repr.kind == 'subray_sum'
        assert c.subrays and all(s.phase_rad is not None for s in c.subrays)


def test_no_native_split_just_delay_clusters():
    """同时延多峰也不分裂 (B-1 烘任意形状) → 1 簇。"""
    mpcs = _grp(5, 6, 0.0) + _grp(95, 6, 0.0)        # F3 会分裂的多峰; F5 不分
    p = phase_continuous_fit(mpcs, V, FC, PROF)
    assert len(p.snapshots[0].clusters) == 1


def test_gcm_native_target():
    mpcs = _grp(30, 5, 0.0)
    p = phase_continuous_fit(mpcs, V, FC, PROF, target_path='GCM_native')
    assert p.clustering_meta.target_path == 'GCM_native'
    assert p.snapshots[0].clusters[0].doppler_repr.kind == 'subray_sum'


# ───────────── fail-loud ─────────────

def test_missing_phase_rad_raises():
    mpcs = [MPC(0.0, 1.0, 30, -30, phase_rad=None)]   # 缺确定性相位
    with pytest.raises(ValueError, match='phase_rad'):
        phase_continuous_fit(mpcs, V, FC, PROF)


def test_b2_target_rejected():
    mpcs = _grp(30, 5, 0.0)
    with pytest.raises(ValueError, match='B1_baked / GCM_native'):
        phase_continuous_fit(mpcs, V, FC, PROF, target_path='B2_parametric')


def test_empty_raises():
    with pytest.raises(ValueError, match='不能为空'):
        phase_continuous_fit([], V, FC, PROF)


# ───────────── budget / 确定性 ─────────────

def test_tap_budget_enforced():
    mpcs = []
    for i in range(8):
        mpcs += _grp(20 + i * 8, 2, delay=i * 100e-9)
    p = phase_continuous_fit(mpcs, V, FC, F64CapabilityProfile(tap_budget=3))
    assert len(p.snapshots[0].clusters) <= 3


def test_deterministic():
    mpcs = _grp(33, 5, 0.0) + _grp(77, 5, 150e-9)
    p1 = phase_continuous_fit(mpcs, V, FC, PROF)
    p2 = phase_continuous_fit(mpcs, V, FC, PROF)
    assert p1.model_dump() == p2.model_dump()
