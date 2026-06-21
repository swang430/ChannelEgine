"""PR-4 — B-2 per-tap 参数表提取 extract_tap_parameters (P2-14)。

native/dual_gaussian 簇 → TapParameter; target_path 守门; 双高斯加权质心; 多快照 warn。
.tap 字节序列化 (字段顺序/gaussian 关键字) 现场标定 (§9.1), 不在本测试范围。
"""
import numpy as np
import pytest

from mimo_ota_simulator.annotated_cdl_schema import (
    AnnotatedCDLProfile, CDLSnapshot, AnnotatedCluster, ClusteringMeta, TrajectoryMeta,
    DopplerNative, DopplerDualGaussian, DopplerBaked, GaussianBeam,
)
from mimo_ota_simulator.b2_tap_params import (
    extract_tap_parameters, TapParameter, _dual_gaussian_centroid_hz,
)


def _native_cluster(cid=0):
    return AnnotatedCluster(
        cluster_id=cid, delay_s=0.0, power_linear=1.0, aoa_deg=30.0, aod_deg=10.0,
        as_aoa_deg=5.0, as_aod_deg=2.0, xpr_db=10.0,
        doppler_repr=DopplerNative(shape='classical', f_d_centroid_hz=50.0,
                                   f_d_max_hz=100.0, native_fit_residual=0.02))


def _dual_cluster(cid=1):
    return AnnotatedCluster(
        cluster_id=cid, delay_s=100e-9, power_linear=0.5, aoa_deg=120.0, aod_deg=80.0,
        doppler_repr=DopplerDualGaussian(
            beam_a=GaussianBeam(center_shift_hz=80.0, std_hz=20.0),
            beam_b=GaussianBeam(center_shift_hz=-40.0, std_hz=30.0),
            power_ratio_db=3.0))


def _b2_profile(clusters, snapshots=None):
    return AnnotatedCDLProfile(
        center_frequency_hz=3.5e9, pathloss_db=80.0, is_los=False,
        clustering_meta=ClusteringMeta(
            algorithm='geometric_native_fit', target_path='B2_parametric',
            test_class='throughput_psd', f64_profile_ref='t', tap_budget_used=len(clusters)),
        snapshots=snapshots or [CDLSnapshot(
            index=0, ue_velocity_mps=(10.0, 0.0, 0.0), clusters=clusters)])


# ───────────── native ─────────────

def test_native_cluster_to_tap():
    taps = extract_tap_parameters(_b2_profile([_native_cluster()]))
    assert len(taps) == 1
    t = taps[0]
    assert t.doppler_kind == 'native'
    assert t.doppler_shape == 'classical'
    assert t.doppler_centroid_hz == 50.0
    assert t.doppler_spread_hz == 100.0       # f_d_max_hz
    assert t.aoa_deg == 30.0 and t.as_aoa_deg == 5.0
    assert t.xpr_db == 10.0
    assert t.native_fit_residual == 0.02


# ───────────── dual_gaussian ─────────────

def test_dual_gaussian_cluster_to_tap():
    t = extract_tap_parameters(_b2_profile([_dual_cluster()]))[0]
    assert t.doppler_kind == 'dual_gaussian'
    assert t.doppler_shape == 'gaussian'
    assert t.doppler_spread_hz == 30.0        # max(20, 30)
    assert t.doppler_params['power_ratio_db'] == 3.0
    assert t.doppler_params['beam_a']['std_hz'] == 20.0
    assert t.doppler_params['beam_b']['center_shift_hz'] == -40.0


def test_dual_gaussian_centroid_power_weighted():
    """质心 = 功率加权 beam center (power_ratio_db=3 → P(a)/P(b)=10^0.3)。"""
    c = _dual_gaussian_centroid_hz(
        GaussianBeam(center_shift_hz=80.0, std_hz=20.0),
        GaussianBeam(center_shift_hz=-40.0, std_hz=30.0), 3.0)
    p_a = 10 ** 0.3
    assert np.isclose(c, (p_a * 80.0 - 40.0) / (p_a + 1.0))


# ───────────── 混合 + 顺序 ─────────────

def test_mixed_native_and_dual_preserves_order():
    taps = extract_tap_parameters(_b2_profile([_native_cluster(0), _dual_cluster(1)]))
    assert len(taps) == 2
    assert taps[0].doppler_kind == 'native'
    assert taps[1].doppler_kind == 'dual_gaussian'
    assert [t.tap_index for t in taps] == [0, 1]


# ───────────── fail-loud / warn ─────────────

def test_rejects_b1_target():
    """target_path=B1_baked → fail-loud (B-1 走烘焙路, 非参数表)。"""
    b1 = AnnotatedCDLProfile(
        center_frequency_hz=3.5e9, pathloss_db=80.0, is_los=False,
        clustering_meta=ClusteringMeta(
            algorithm='raw_passthrough', target_path='B1_baked',
            test_class='throughput_psd', f64_profile_ref='t', tap_budget_used=1),
        snapshots=[CDLSnapshot(index=0, ue_velocity_mps=(10.0, 0.0, 0.0), clusters=[
            AnnotatedCluster(cluster_id=0, delay_s=0.0, power_linear=1.0, aoa_deg=30.0,
                aod_deg=10.0, doppler_repr=DopplerBaked())])])
    with pytest.raises(ValueError, match="B2_parametric"):
        extract_tap_parameters(b1)


def test_multisnapshot_warns_and_extracts_one():
    snaps = [CDLSnapshot(index=0, ue_velocity_mps=(10.0, 0.0, 0.0), clusters=[_native_cluster()]),
             CDLSnapshot(index=1, ue_velocity_mps=(10.0, 0.0, 0.0), clusters=[_native_cluster()])]
    prof = AnnotatedCDLProfile(
        center_frequency_hz=3.5e9, pathloss_db=80.0, is_los=False,
        clustering_meta=ClusteringMeta(
            algorithm='geometric_native_fit', target_path='B2_parametric',
            test_class='throughput_psd', f64_profile_ref='t', tap_budget_used=1),
        trajectory_meta=TrajectoryMeta(
            n_snapshots=2, skeleton_rate_hz=100.0, delta_d_geo_m=0.04, route_len_m=1.0),
        snapshots=snaps)
    with pytest.warns(UserWarning, match="单快照"):
        taps = extract_tap_parameters(prof)
    assert len(taps) == 1


def test_snapshot_index_out_of_range():
    with pytest.raises(IndexError, match="不在 profile"):
        extract_tap_parameters(_b2_profile([_native_cluster()]), snapshot_index=5)
