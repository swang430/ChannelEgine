"""PR-3 — §6 路径判决 select_path_and_clustering (P2-14)。

覆盖 test_class 四分支 + GCM 可用性门 + ESCALATE fail-loud (绝不静默回落)。
"""
import numpy as np
import pytest

from mimo_ota_simulator.geometric_native_fit import MPC, F64CapabilityProfile
from mimo_ota_simulator.path_decision import (
    select_path_and_clustering, f_d_max_hz, PathDecision,
)

V_LOW = (10.0, 0.0, 0.0)       # |v|=10 m/s
FC = 3.5e9


def _simple_mpcs():
    """单簇低 AS (native 可表示): 几条射线集中在一个角度。"""
    return [
        MPC(delay_s=0.0, power_linear=1.0, aoa_deg=30.0, aod_deg=10.0),
        MPC(delay_s=1e-9, power_linear=0.8, aoa_deg=31.0, aod_deg=10.5),
        MPC(delay_s=2e-9, power_linear=0.6, aoa_deg=29.0, aod_deg=9.5),
    ]


# ───────────── f_D,max ─────────────

def test_f_d_max_formula():
    # |v|·f_c/c = 10 * 3.5e9 / 3e8 = 116.67 Hz
    assert np.isclose(f_d_max_hz(V_LOW, FC), 10 * 3.5e9 / 3e8)
    # 与方向无关 (取模)
    assert np.isclose(f_d_max_hz((0.0, 10.0, 0.0), FC), f_d_max_hz(V_LOW, FC))


# ───────────── 吞吐/一致性 → B-2 普及 ─────────────

def test_throughput_native_representable_is_b2():
    d = select_path_and_clustering('throughput_psd', _simple_mpcs(), V_LOW, FC)
    assert d.target_path == 'B2_parametric'
    assert d.clustering_algo == 'geometric_native_fit'
    assert not d.is_escalation


def test_throughput_native_unrepresentable_falls_back_b1():
    """native 残差超阈 (分散 doppler + budget 满不能分裂) → B-1 退路 (非 B-2)。"""
    # 5 射线同时延、aoa 0/45/90/135/180 → doppler 多模态分散; tap_budget=1 不分裂 → resid 大
    disp = [MPC(delay_s=0.0, power_linear=1.0, aoa_deg=a, aod_deg=0.0)
            for a in (0.0, 45.0, 90.0, 135.0, 180.0)]
    f64 = F64CapabilityProfile(tap_budget=1, rho_thresh=0.01)   # resid≈0.018 > 0.01
    d = select_path_and_clustering('throughput_psd', disp, (60.0, 0.0, 0.0), FC, f64)
    assert d.target_path == 'B1_baked'
    assert d.clustering_algo == 'phase_continuous'
    assert d.native_fit_residual > f64.rho_thresh


def test_consistency_class_also_b2_route():
    d = select_path_and_clustering('consistency', _simple_mpcs(), V_LOW, FC)
    assert d.target_path in ('B2_parametric', 'B1_baked')   # 走 B-2 普及判决 (非确定性相位路)
    assert d.clustering_algo in ('geometric_native_fit', 'phase_continuous')


# ───────────── 确定性相位 (ISAC/波束跟踪) → B-1 / GCM ─────────────

def test_isac_low_centroid_is_b1():
    """低 f_D,max 确定性相位 → B-1 可行 (非 GCM)。"""
    f64 = F64CapabilityProfile(large_centroid_thresh_hz=1e6)   # 高阈, 116Hz 远低
    d = select_path_and_clustering('isac_sensing', _simple_mpcs(), V_LOW, FC, f64)
    assert d.target_path == 'B1_baked'
    assert d.clustering_algo == 'phase_continuous'


def test_isac_large_centroid_with_gcm_is_gcm():
    """大质心 + 有 GCM license → GCM。"""
    f64 = F64CapabilityProfile(large_centroid_thresh_hz=100.0, has_gcm=True)  # 116>100
    d = select_path_and_clustering('isac_sensing', _simple_mpcs(), V_LOW, FC, f64)
    assert d.target_path == 'GCM_native'
    assert d.clustering_algo == 'phase_continuous'


def test_isac_large_centroid_without_gcm_escalates():
    """大质心 + 无 GCM → ESCALATE fail-loud, 绝不静默回落 B1_baked (Codex P2 #165 母题)。"""
    f64 = F64CapabilityProfile(large_centroid_thresh_hz=100.0, has_gcm=False)
    d = select_path_and_clustering('beam_tracking', _simple_mpcs(), V_LOW, FC, f64)
    assert d.target_path == 'ESCALATE'
    assert d.is_escalation
    assert d.target_path != 'B1_baked'        # 显式断言不回落


def test_fr2_high_speed_without_gcm_escalates():
    """FR2 (28GHz) + 高速 (50 m/s) + 无 GCM → ESCALATE。"""
    f64 = F64CapabilityProfile(has_gcm=False)   # 默认 fr2=24e9, 高速门 30 m/s
    d = select_path_and_clustering('beam_tracking', _simple_mpcs(), (50.0, 0.0, 0.0),
                                   28e9, f64)
    assert d.target_path == 'ESCALATE'


def test_fr2_high_speed_with_gcm_is_gcm():
    f64 = F64CapabilityProfile(has_gcm=True)
    d = select_path_and_clustering('isac_sensing', _simple_mpcs(), (50.0, 0.0, 0.0),
                                   28e9, f64)
    assert d.target_path == 'GCM_native'


# ───────────── fail-loud ─────────────

def test_empty_mpcs_raises():
    with pytest.raises(ValueError, match="mpcs 为空"):
        select_path_and_clustering('throughput_psd', [], V_LOW, FC)
