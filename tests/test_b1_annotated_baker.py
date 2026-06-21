"""PR-1b — B-1 标注式烘焙器 bake_b1_annotated (AnnotatedCDLProfile → .asc, OTA)。

验证: subray_sum / baked 簇都能烘出 per-(Tx,Probe) .asc; OTA 校准真施加 (不当黑盒);
target_path=B2 fail-loud; 多快照 warn。.asc 字节格式现成 (PropsimASCIIExporter), 零现场。
"""
import os

import numpy as np
import pytest

from mimo_ota_simulator.b1_annotated_baker import bake_b1_annotated
from mimo_ota_simulator.annotated_cdl_schema import (
    AnnotatedCDLProfile, CDLSnapshot, AnnotatedCluster, ClusteringMeta,
    TrajectoryMeta, DopplerSubraySum, DopplerBaked, DopplerNative, SubRay,
)
from mimo_ota_simulator.data_models import (
    ChamberConfig, TargetChannelConfig, AntennaArrayConfig,
    PropsimExportConfig, CalibrationConfig, CalibrationEntry,
)

N_PROBES = 4
N_TX = 2


def _chamber():
    return ChamberConfig(num_probes=N_PROBES, dual_polarized=False,
                         distribution='ring', radius_m=1.0)


def _channel():
    return TargetChannelConfig(
        model_name='UMi-StreetCanyon', cluster_model_name='CDL-A',
        center_frequency_hz=3.5e9, input_mode='custom',
        tx_antenna=AntennaArrayConfig(num_cols=N_TX, array_type='ULA'),
        rx_antenna=AntennaArrayConfig(num_cols=2, array_type='ULA'),
    )


def _export(tmp_path, name='b1.asc'):
    return PropsimExportConfig(filename=str(tmp_path / name), duration_s=1e-4,
                              sample_rate_hz=1e6)


def _subray_cluster(cid=0):
    return AnnotatedCluster(
        cluster_id=cid, delay_s=0.0, power_linear=1.0,
        aoa_deg=20.0, aod_deg=10.0, zoa_deg=90.0, zod_deg=90.0,
        as_aoa_deg=5.0, as_aod_deg=2.0,
        doppler_repr=DopplerSubraySum(f_d_centroid_hz=10.0),
        subrays=[
            SubRay(aoa_deg=20.0, aod_deg=10.0, power_linear=1.0, phase_rad=0.5),
            SubRay(aoa_deg=22.0, aod_deg=11.0, power_linear=0.5, phase_rad=1.2),
        ],
    )


def _baked_cluster(cid=0):
    return AnnotatedCluster(
        cluster_id=cid, delay_s=0.0, power_linear=1.0,
        aoa_deg=20.0, aod_deg=10.0, zoa_deg=90.0, zod_deg=90.0,
        as_aoa_deg=5.0, as_aod_deg=2.0,
        doppler_repr=DopplerBaked(), physicality='standard_fan',
    )


def _annotated(cluster, algorithm='phase_continuous', snapshots=None):
    return AnnotatedCDLProfile(
        center_frequency_hz=3.5e9, pathloss_db=80.0, is_los=False,
        clustering_meta=ClusteringMeta(
            algorithm=algorithm, target_path='B1_baked',
            test_class='throughput_psd', f64_profile_ref='test', tap_budget_used=1,
        ),
        snapshots=snapshots or [CDLSnapshot(
            index=0, ue_velocity_mps=(10.0, 0.0, 0.0), clusters=[cluster])],
    )


# ───────────── 烘焙产出 .asc ─────────────

def test_bake_subray_sum_produces_asc(tmp_path):
    files = bake_b1_annotated(_annotated(_subray_cluster()), _chamber(), _channel(),
                              _export(tmp_path))
    assert len(files) == N_PROBES * N_TX          # per Tx→Probe link
    assert all(os.path.exists(f) and os.path.getsize(f) > 0 for f in files)


def test_bake_baked_cluster_produces_asc(tmp_path):
    files = bake_b1_annotated(_annotated(_baked_cluster(), algorithm='raw_passthrough'),
                              _chamber(), _channel(), _export(tmp_path))
    assert len(files) == N_PROBES * N_TX
    assert all(os.path.getsize(f) > 0 for f in files)


# ───────────── OTA 校准真施加 (不当黑盒) ─────────────

def test_calibration_actually_applied(tmp_path):
    """非平凡 OTA 校准 (per-probe gain 差异) 改变 .asc 内容; None 校准则不施加。
    断言"有校准 vs 无校准"的 .asc 字节不同 → 校准确实进了导出 (V1.0 §8.1)。"""
    annotated = _annotated(_subray_cluster())
    cal = CalibrationConfig(entries=[
        CalibrationEntry(port_id=1, probe_gain_dbi=10.0, cable_phase_deg=30.0),
        CalibrationEntry(port_id=2, probe_gain_dbi=-6.0, cable_phase_deg=-45.0),
        CalibrationEntry(port_id=3, probe_gain_dbi=3.0),
        CalibrationEntry(port_id=4, probe_gain_dbi=-3.0),
    ])
    f_cal = bake_b1_annotated(annotated, _chamber(), _channel(),
                              _export(tmp_path, 'cal.asc'), calibration_config=cal)
    f_nocal = bake_b1_annotated(annotated, _chamber(), _channel(),
                                _export(tmp_path, 'nocal.asc'), calibration_config=None)
    cal_txt = open(f_cal[0]).read()
    nocal_txt = open(f_nocal[0]).read()
    assert cal_txt != nocal_txt                   # 校准预失真改变了导出幅度/相位


# ───────────── fail-loud ─────────────

def test_rejects_b2_parametric_target(tmp_path):
    """target_path=B2_parametric → fail-loud (B-2 走 .tap 参数化路, 非本烘焙器)。"""
    b2 = AnnotatedCDLProfile(
        center_frequency_hz=3.5e9, pathloss_db=80.0, is_los=False,
        clustering_meta=ClusteringMeta(
            algorithm='geometric_native_fit', target_path='B2_parametric',
            test_class='throughput_psd', f64_profile_ref='t', tap_budget_used=1),
        snapshots=[CDLSnapshot(index=0, ue_velocity_mps=(10.0, 0.0, 0.0), clusters=[
            AnnotatedCluster(cluster_id=0, delay_s=0.0, power_linear=1.0,
                aoa_deg=20.0, aod_deg=10.0,
                doppler_repr=DopplerNative(shape='classical', f_d_max_hz=50.0))])],
    )
    with pytest.raises(ValueError, match="B1_baked|B2_parametric"):
        bake_b1_annotated(b2, _chamber(), _channel(), _export(tmp_path))


def test_snapshot_index_out_of_range(tmp_path):
    with pytest.raises(IndexError, match="snapshot_index"):
        bake_b1_annotated(_annotated(_subray_cluster()), _chamber(), _channel(),
                          _export(tmp_path), snapshot_index=5)


# ───────────── 多快照 warn ─────────────

def test_multisnapshot_warns_and_bakes_one(tmp_path):
    """多快照 (动态轨迹) → warn 单快照先行, 仍烘出 snapshot_index 的 .asc。"""
    snaps = [
        CDLSnapshot(index=0, ue_velocity_mps=(10.0, 0.0, 0.0), clusters=[_subray_cluster()]),
        CDLSnapshot(index=1, ue_velocity_mps=(10.0, 0.0, 0.0), clusters=[_subray_cluster()]),
    ]
    annotated = AnnotatedCDLProfile(
        center_frequency_hz=3.5e9, pathloss_db=80.0, is_los=False,
        clustering_meta=ClusteringMeta(
            algorithm='phase_continuous', target_path='B1_baked',
            test_class='throughput_psd', f64_profile_ref='t', tap_budget_used=1),
        trajectory_meta=TrajectoryMeta(
            n_snapshots=2, skeleton_rate_hz=100.0, delta_d_geo_m=0.04, route_len_m=1.0),
        snapshots=snaps,
    )
    with pytest.warns(UserWarning, match="单快照"):
        files = bake_b1_annotated(annotated, _chamber(), _channel(), _export(tmp_path))
    assert len(files) == N_PROBES * N_TX


def test_selects_snapshot_by_declared_index_not_list_offset():
    """Codex P2: snapshots 列表乱序 [index=1, index=0] 时 snapshot_index=0 取 declared
    index=0 的快照 (非 list offset 0 = index=1)。schema 仅保证 index 集合连续、不保证
    list 顺序。用 ue_velocity 区分两快照。"""
    from mimo_ota_simulator.b1_annotated_baker import _select_snapshot
    s0 = CDLSnapshot(index=0, ue_velocity_mps=(1.0, 0.0, 0.0), clusters=[_subray_cluster()])
    s1 = CDLSnapshot(index=1, ue_velocity_mps=(2.0, 0.0, 0.0), clusters=[_subray_cluster()])
    annotated = AnnotatedCDLProfile(
        center_frequency_hz=3.5e9, pathloss_db=80.0, is_los=False,
        clustering_meta=ClusteringMeta(
            algorithm='phase_continuous', target_path='B1_baked',
            test_class='throughput_psd', f64_profile_ref='t', tap_budget_used=1),
        trajectory_meta=TrajectoryMeta(
            n_snapshots=2, skeleton_rate_hz=100.0, delta_d_geo_m=0.04, route_len_m=1.0),
        snapshots=[s1, s0],     # list 乱序: index=1 在前
    )
    assert _select_snapshot(annotated, 0).ue_velocity_mps == (1.0, 0.0, 0.0)   # declared 0
    assert _select_snapshot(annotated, 1).ue_velocity_mps == (2.0, 0.0, 0.0)   # declared 1
    with pytest.raises(IndexError, match="不在 profile"):
        _select_snapshot(annotated, 9)
