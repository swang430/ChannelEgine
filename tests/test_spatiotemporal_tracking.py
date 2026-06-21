"""F4 单测 — 时空跟踪层 (P2-14)。"""
import numpy as np
import pytest

from mimo_ota_simulator.geometric_native_fit import F64CapabilityProfile, MPC
from mimo_ota_simulator.spatiotemporal_tracking import native_fit_trajectory

V = [40.0, 0.0, 0.0]
FC = 3.5e9
PROF = F64CapabilityProfile(tap_budget=24, rho_thresh=0.06)
TRAJ = dict(skeleton_rate_hz=100.0, delta_d_geo_m=1.0, route_len_m=100.0)


def _grp(aoa, n, delay):
    return [MPC(delay_s=delay, power_linear=1.0, aoa_deg=aoa + o, aod_deg=-aoa)
            for o in np.linspace(-2, 2, n)]


def _ids(snap):
    return {c.cluster_id for c in snap.clusters}


# ───────────── 持续 / 跟踪一致性 ─────────────

def test_persistent_cluster_keeps_same_id():
    """同一簇小幅漂移 (30°→32°) → 两快照保持同 cluster_id。"""
    snaps = [_grp(30, 6, 0.0), _grp(32, 6, 0.0)]
    p = native_fit_trajectory(snaps, V, FC, PROF, **TRAJ)
    assert len(p.snapshots) == 2
    assert _ids(p.snapshots[0]) == _ids(p.snapshots[1])          # 身份一致 (无跳变)
    # 该簇全程 stable
    for snap in p.snapshots:
        assert snap.clusters[0].power_ramp == 'stable'


# ───────────── 生 / 灭 ─────────────

def test_birth_mid_trajectory():
    """快照1 多出一个新簇 (90°@200ns) → birth (birth_index=1, ramp=birth)。"""
    snaps = [_grp(30, 6, 0.0), _grp(30, 6, 0.0) + _grp(90, 6, 200e-9)]
    p = native_fit_trajectory(snaps, V, FC, PROF, **TRAJ)
    born = [c for c in p.snapshots[1].clusters if c.power_ramp == 'birth']
    assert born and all(c.birth_index == 1 for c in born)
    assert len(_ids(p.snapshots[1])) == len(_ids(p.snapshots[0])) + 1


def test_death_mid_trajectory():
    """快照0 的 90° 簇在快照1 消失 → death (death_index=0, ramp=death)。"""
    snaps = [_grp(30, 6, 0.0) + _grp(90, 6, 200e-9), _grp(30, 6, 0.0)]
    p = native_fit_trajectory(snaps, V, FC, PROF, **TRAJ)
    dead = [c for c in p.snapshots[0].clusters if c.power_ramp == 'death']
    assert dead and all(c.death_index == 0 for c in dead)


def test_stable_cluster_throughout_three_snapshots():
    """簇全程在 → 始终 stable (非生灭)。"""
    snaps = [_grp(40, 6, 0.0), _grp(41, 6, 0.0), _grp(42, 6, 0.0)]
    p = native_fit_trajectory(snaps, V, FC, PROF, **TRAJ)
    for snap in p.snapshots:
        assert all(c.power_ramp == 'stable' for c in snap.clusters)
    # 三快照同一 id
    assert _ids(p.snapshots[0]) == _ids(p.snapshots[1]) == _ids(p.snapshots[2])


# ───────────── 多快照 profile 合法性 ─────────────

def test_dynamic_profile_valid():
    snaps = [_grp(30, 6, 0.0), _grp(33, 6, 0.0), _grp(36, 6, 0.0)]
    p = native_fit_trajectory(snaps, V, FC, PROF, **TRAJ)
    assert p.trajectory_meta is not None
    assert p.trajectory_meta.n_snapshots == 3
    assert [s.index for s in p.snapshots] == [0, 1, 2]            # 连续 0..n-1
    assert p.clustering_meta.target_path == 'B2_parametric'


def test_per_snapshot_velocity():
    snaps = [_grp(30, 6, 0.0), _grp(30, 6, 0.0)]
    vels = [[40, 0, 0], [30, 0, 0]]
    p = native_fit_trajectory(snaps, vels, FC, PROF, **TRAJ)
    assert p.snapshots[0].ue_velocity_mps == (40, 0, 0)
    assert p.snapshots[1].ue_velocity_mps == (30, 0, 0)


# ───────────── 边缘 / 确定性 ─────────────

def test_empty_raises():
    with pytest.raises(ValueError, match='不能为空'):
        native_fit_trajectory([], V, FC, PROF, **TRAJ)


def test_deterministic():
    snaps = [_grp(30, 6, 0.0), _grp(33, 6, 0.0)]
    p1 = native_fit_trajectory(snaps, V, FC, PROF, **TRAJ)
    p2 = native_fit_trajectory(snaps, V, FC, PROF, **TRAJ)
    assert p1.model_dump() == p2.model_dump()
