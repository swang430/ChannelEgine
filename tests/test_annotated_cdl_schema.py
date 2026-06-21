"""F2 单测 — 标注式 CDL schema + §5.4 一致性校验 (P2-14)。"""
import pytest
from pydantic import ValidationError

from mimo_ota_simulator.annotated_cdl_schema import (
    AnnotatedCDLProfile,
    AnnotatedCluster,
    CDLSnapshot,
    ClusteringMeta,
    DopplerBaked,
    DopplerDualGaussian,
    DopplerNative,
    DopplerSubraySum,
    TrajectoryMeta,
)
from mimo_ota_simulator.cdl_schema import CustomCDLProfile


# ───────────── 构造 helper ─────────────

def _native_cluster(cid=0, **kw):
    base = dict(cluster_id=cid, delay_s=0.0, power_linear=1.0, aoa_deg=10.0, aod_deg=-20.0,
                doppler_repr=DopplerNative(shape='gaussian', f_d_centroid_hz=200.0,
                                           f_d_max_hz=40.0, shape_params={'std_hz': 30.0},
                                           native_fit_residual=0.01))
    base.update(kw)
    return AnnotatedCluster(**base)


def _subray_cluster(cid=0, with_phase=True):
    from mimo_ota_simulator.annotated_cdl_schema import SubRay
    rays = [SubRay(aoa_deg=10, aod_deg=-20, power_linear=0.5,
                   phase_rad=(0.1 if with_phase else None)) for _ in range(2)]
    return AnnotatedCluster(cluster_id=cid, delay_s=0.0, power_linear=1.0, aoa_deg=10, aod_deg=-20,
                            doppler_repr=DopplerSubraySum(), subrays=rays)


def _profile(clusters, *, target_path='B2_parametric', algorithm='geometric_native_fit',
             test_class='throughput_psd', trajectory_meta=None, snapshots=None):
    snaps = snapshots or [CDLSnapshot(index=0, ue_velocity_mps=(30, 0, 0), clusters=clusters)]
    return AnnotatedCDLProfile(
        center_frequency_hz=3.5e9, pathloss_db=98.0, is_los=False,
        clustering_meta=ClusteringMeta(algorithm=algorithm, target_path=target_path,
                                       test_class=test_class, f64_profile_ref='test',
                                       tap_budget_used=len(clusters)),
        trajectory_meta=trajectory_meta,
        snapshots=snaps)


# ───────────── 合法构造 ─────────────

def test_valid_b2_profile():
    p = _profile([_native_cluster(0),
                  _native_cluster(1, doppler_repr=DopplerDualGaussian(
                      beam_a={'center_shift_hz': -100, 'std_hz': 30},
                      beam_b={'center_shift_hz': 80, 'std_hz': 25}, power_ratio_db=3.0))])
    assert p.clustering_meta.target_path == 'B2_parametric'
    assert p.snapshots[0].clusters[1].doppler_repr.kind == 'dual_gaussian'


def test_valid_b1_subray_sum_profile():
    p = _profile([_subray_cluster(0)], target_path='B1_baked', algorithm='phase_continuous')
    assert p.snapshots[0].clusters[0].doppler_repr.kind == 'subray_sum'


def test_discriminated_union_from_dict():
    """doppler_repr 按 kind 反序列化到正确子类型。"""
    p = AnnotatedCDLProfile.model_validate({
        'center_frequency_hz': 3.5e9, 'pathloss_db': 98.0, 'is_los': False,
        'clustering_meta': {'algorithm': 'geometric_native_fit', 'target_path': 'B2_parametric',
                            'test_class': 'throughput_psd', 'f64_profile_ref': 't', 'tap_budget_used': 1},
        'snapshots': [{'index': 0, 'ue_velocity_mps': [30, 0, 0], 'clusters': [
            {'cluster_id': 0, 'delay_s': 0.0, 'power_linear': 1.0, 'aoa_deg': 10, 'aod_deg': -20,
             'doppler_repr': {'kind': 'native', 'shape': 'classical', 'f_d_max_hz': 350.0}}]}]})
    dr = p.snapshots[0].clusters[0].doppler_repr
    assert isinstance(dr, DopplerNative) and dr.shape == 'classical'


# ───────────── §5.4 一致性校验 (反向) ─────────────

def test_b2_path_rejects_baked_cluster():
    with pytest.raises(ValidationError, match='不允许 doppler_repr.kind=baked'):
        _profile([AnnotatedCluster(cluster_id=0, delay_s=0, power_linear=1.0, aoa_deg=1, aod_deg=1,
                                   doppler_repr=DopplerBaked())], target_path='B2_parametric')


def test_b1_path_rejects_native_cluster():
    with pytest.raises(ValidationError, match='不允许 doppler_repr.kind=native'):
        _profile([_native_cluster(0)], target_path='B1_baked', algorithm='raw_passthrough')


def test_subray_sum_requires_phase_rad():
    with pytest.raises(ValidationError, match='phase_rad'):
        _profile([_subray_cluster(0, with_phase=False)], target_path='B1_baked',
                 algorithm='phase_continuous')


def test_phase_continuous_rejects_b2():
    with pytest.raises(ValidationError, match='phase_continuous .*不能.*B2'):
        _profile([_subray_cluster(0)], target_path='B2_parametric', algorithm='phase_continuous')


def test_duplicate_cluster_id_in_snapshot_raises():
    with pytest.raises(ValidationError, match='cluster_id=0 重复'):
        _profile([_native_cluster(0), _native_cluster(0)])


def test_trajectory_nsnapshots_mismatch_raises():
    with pytest.raises(ValidationError, match='n_snapshots'):
        _profile([_native_cluster(0)],
                 trajectory_meta=TrajectoryMeta(n_snapshots=5, skeleton_rate_hz=100,
                                                delta_d_geo_m=1.0, route_len_m=100))


# ───────────── Codex P2 #35 修复 ─────────────

def test_dual_gaussian_beam_fields_validated():
    """beam 字段受校验: 缺 std_hz / std_hz<=0 → raises (Codex P2)。"""
    with pytest.raises(ValidationError):
        DopplerDualGaussian(beam_a={'center_shift_hz': 0.0},                 # 缺 std_hz
                            beam_b={'center_shift_hz': 0.0, 'std_hz': 10.0})
    with pytest.raises(ValidationError):
        DopplerDualGaussian(beam_a={'center_shift_hz': 0.0, 'std_hz': -1.0}, # std_hz<=0
                            beam_b={'center_shift_hz': 0.0, 'std_hz': 10.0})


def _snap(idx, cid=5, v=(30, 0, 0)):
    return CDLSnapshot(index=idx, ue_velocity_mps=v, clusters=[_native_cluster(cid)])


def test_multi_snapshot_requires_trajectory_meta():
    """多快照但缺 trajectory_meta → raises (Codex P2)。"""
    with pytest.raises(ValidationError, match='trajectory_meta'):
        _profile([_native_cluster(5)], snapshots=[_snap(0), _snap(1)])  # trajectory_meta=None


def test_duplicate_snapshot_index_raises():
    """快照 index 重复 → raises (Codex P2)。"""
    tm = TrajectoryMeta(n_snapshots=2, skeleton_rate_hz=100, delta_d_geo_m=1.0, route_len_m=100)
    with pytest.raises(ValidationError, match='index 必须是连续'):
        _profile([_native_cluster(5)], snapshots=[_snap(0), _snap(0)], trajectory_meta=tm)


def test_snapshot_index_gap_raises():
    """快照 index 跳号 (0,2 缺 1) → raises (Codex P2: gaps 也歧义)。"""
    tm = TrajectoryMeta(n_snapshots=2, skeleton_rate_hz=100, delta_d_geo_m=1.0, route_len_m=100)
    with pytest.raises(ValidationError, match='无跳号'):
        _profile([_native_cluster(5)], snapshots=[_snap(0), _snap(2)], trajectory_meta=tm)


def test_valid_dynamic_multi_snapshot():
    """合法动态: 2 快照 + 唯一 index + trajectory_meta 一致 → OK, 跨快照同 cluster_id (连续性)。"""
    tm = TrajectoryMeta(n_snapshots=2, skeleton_rate_hz=100, delta_d_geo_m=1.0, route_len_m=100)
    p = _profile([_native_cluster(5)], snapshots=[_snap(0), _snap(1, v=(31, 0, 0))],
                 trajectory_meta=tm)
    assert len(p.snapshots) == 2
    assert p.snapshots[1].clusters[0].cluster_id == 5


# ───────────── 向后兼容 ─────────────

def test_from_custom_profile_roundtrips_to_b1_baked():
    custom = CustomCDLProfile.from_dict({
        'center_frequency_hz': 3.5e9, 'pathloss_db': 100.0, 'is_los': True,
        'ue_velocity_mps': [30, 0, 0], 'k_factor_db': 7.0,
        'clusters': [
            {'delay_s': 0.0, 'power_linear': 0.6, 'aoa_deg': 10, 'aod_deg': -20, 'xpr_db': 8.0},
            {'delay_s': 1e-7, 'power_linear': 0.4, 'aoa_deg': 45, 'aod_deg': 30}]})
    ann = AnnotatedCDLProfile.from_custom_profile(custom)
    assert ann.clustering_meta.target_path == 'B1_baked'
    assert ann.clustering_meta.algorithm == 'raw_passthrough'
    assert ann.clustering_meta.physicality == 'standard_fan'
    assert len(ann.snapshots) == 1
    assert len(ann.snapshots[0].clusters) == 2
    assert all(c.doppler_repr.kind == 'baked' for c in ann.snapshots[0].clusters)
    assert ann.snapshots[0].clusters[0].cluster_id == 0
    assert ann.k_factor_db == 7.0


# ───────────── round-trip ─────────────

def test_json_roundtrip():
    p = _profile([_native_cluster(0)])
    p2 = AnnotatedCDLProfile.model_validate(p.model_dump())
    assert p2 == p
    assert p2.snapshots[0].clusters[0].doppler_repr.kind == 'native'
