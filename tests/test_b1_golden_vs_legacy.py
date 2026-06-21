"""PR-2 — B-1 金标准软件对照 (§10.1 软件半, P2-14)。

验证新架构 B-1 baked 路 (from_custom_profile → bake_b1_annotated) 与现有直接路径
(CustomCDLBuilder + synthesize_ota + PropsimASCIIExporter) 产出【逐位一致】(同 seed),
确认新标注式架构没给既有 B-1 引入回归。

QZ 处 Doppler PSD / 空间相关 / 衰落 CDF 实测对照 (标定 rho_thresh) 属现场半 (§10.1),
不在本软件对照范围。
"""
import numpy as np

from mimo_ota_simulator.cdl_schema import CustomCDLProfile, CDLClusterSpec
from mimo_ota_simulator.channel_builders import CustomCDLBuilder
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.exporters import PropsimASCIIExporter
from mimo_ota_simulator.annotated_cdl_schema import AnnotatedCDLProfile
from mimo_ota_simulator.b1_annotated_baker import bake_b1_annotated, _snapshot_to_custom
from mimo_ota_simulator.data_models import (
    ChamberConfig, TargetChannelConfig, AntennaArrayConfig, PropsimExportConfig,
)

SEED = 42


def _custom():
    return CustomCDLProfile(
        center_frequency_hz=3.5e9, pathloss_db=80.0, is_los=False,
        ue_velocity_mps=(10.0, 0.0, 0.0),
        clusters=[
            CDLClusterSpec(delay_s=0.0, power_linear=1.0, aoa_deg=30.0, aod_deg=10.0,
                           as_aoa_deg=5.0, as_aod_deg=2.0),
            CDLClusterSpec(delay_s=100e-9, power_linear=0.5, aoa_deg=120.0, aod_deg=80.0,
                           as_aoa_deg=8.0, as_aod_deg=3.0),
        ],
    )


def _chamber():
    return ChamberConfig(num_probes=4, dual_polarized=False, distribution='ring', radius_m=1.0)


def _channel():
    return TargetChannelConfig(
        model_name='UMi-StreetCanyon', cluster_model_name='CDL-A',
        center_frequency_hz=3.5e9, input_mode='custom', random_seed=SEED,
        tx_antenna=AntennaArrayConfig(num_cols=2, array_type='ULA'),
        rx_antenna=AntennaArrayConfig(num_cols=2, array_type='ULA'),
    )


def _export(tmp_path, name):
    return PropsimExportConfig(filename=str(tmp_path / name), duration_s=1e-4, sample_rate_hz=1e6)


def test_from_custom_roundtrip_lossless():
    """from_custom_profile → _snapshot_to_custom round-trip 保标准几何无损 (对照基础)。"""
    custom = _custom()
    annotated = AnnotatedCDLProfile.from_custom_profile(custom)
    back = _snapshot_to_custom(annotated, annotated.snapshots[0])
    assert len(back.clusters) == len(custom.clusters)
    for c0, c1 in zip(custom.clusters, back.clusters):
        assert np.isclose(c0.delay_s, c1.delay_s)
        assert np.isclose(c0.power_linear, c1.power_linear)
        assert np.isclose(c0.aoa_deg, c1.aoa_deg)
        assert np.isclose(c0.aod_deg, c1.aod_deg)
        assert np.isclose(c0.as_aoa_deg, c1.as_aoa_deg)
        assert np.isclose(c0.as_zoa_deg, c1.as_zoa_deg)
    assert back.center_frequency_hz == custom.center_frequency_hz
    assert tuple(back.ue_velocity_mps) == tuple(custom.ue_velocity_mps)
    assert back.is_los == custom.is_los


def test_new_b1_baked_matches_legacy_direct(tmp_path):
    """金标准对照: from_custom_profile (baked) → bake_b1_annotated == 现有
    CustomCDLBuilder + synthesize_ota + exporter 直接路径 (同 seed, 逐位一致 .asc)。"""
    custom = _custom()
    chamber, channel = _chamber(), _channel()

    # 路径 A — 现有 B-1 直接 (hardware_pipeline 的 custom CDL 路径); 手动 seed 模拟 run() 播种
    np.random.seed(SEED)
    profile_a = CustomCDLBuilder(custom).build()
    sim_a = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile_a, synthesis_method='ray')
    files_a = PropsimASCIIExporter(_export(tmp_path, 'legacy.asc')).export(sim_a, direction='dl')

    # 路径 B — 新架构 from_custom_profile → bake (内部同 builder+synthesize+exporter + 同 seed 播种)
    annotated = AnnotatedCDLProfile.from_custom_profile(custom)
    files_b = bake_b1_annotated(annotated, chamber, channel, _export(tmp_path, 'new.asc'))

    assert len(files_a) == len(files_b) > 0
    for fa, fb in zip(sorted(files_a), sorted(files_b)):
        assert open(fa).read() == open(fb).read()        # 逐位一致 → 新架构无回归


def test_new_b1_strict_pfs_matches_legacy(tmp_path):
    """strict_pfs 法下 (逐探头独立衰落) 新旧两路也逐位一致。"""
    custom = _custom()
    chamber, channel = _chamber(), _channel()

    np.random.seed(SEED)
    profile_a = CustomCDLBuilder(custom).build()
    sim_a = MIMO_OTA_Simulator().synthesize_ota(
        chamber, channel, profile_a, synthesis_method='strict_pfs')
    files_a = PropsimASCIIExporter(_export(tmp_path, 'legacy_pfs.asc')).export(sim_a, direction='dl')

    annotated = AnnotatedCDLProfile.from_custom_profile(custom)
    files_b = bake_b1_annotated(annotated, chamber, channel, _export(tmp_path, 'new_pfs.asc'),
                                synthesis_method='strict_pfs')

    assert len(files_a) == len(files_b) > 0          # 防 zip 截断掩盖文件数回归 (Codex P2)
    for fa, fb in zip(sorted(files_a), sorted(files_b)):
        assert open(fa).read() == open(fb).read()
