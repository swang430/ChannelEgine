"""PR-1b — B-1 标注式烘焙器: AnnotatedCDLProfile → .asc (P2-14, OTA)。

消费 AnnotatedCDLProfile (target_path=B1_baked / GCM_native):
  - subray_sum 簇 → 真实 RT 子径确定性相干求和 (经 simulator.cluster_subrays 旁路);
  - baked 簇 → 标准 38.901 扇展开 (现行 B-1 行为)。
经 CustomCDLBuilder + MIMO_OTA_Simulator.synthesize_ota 合成 OTA sim_result, 再经
PropsimASCIIExporter 烘 per-(Tx, Probe) .asc (含 OTA 校准预失真)。

边界 (设计 V1.0 §8.1):
  - 本轮 **OTA-only** (探头展开 + OTA 校准 + PAS); conducted 独立条目。
  - 单快照先行 (snapshot_index); 多快照动态轨迹 (跨快照 .asc 帧拼接) 进 backlog。
  - target_path=B2_parametric 不走此路 (走 B-2 .tap 参数化, fail-loud)。

纯软件 (.asc 格式已知在用), 零现场依赖。
"""
import warnings
from typing import Dict, List, Optional

from .annotated_cdl_schema import AnnotatedCDLProfile, CDLSnapshot
from .cdl_schema import CustomCDLProfile, CDLClusterSpec
from .channel_builders import CustomCDLBuilder
from .simulator import MIMO_OTA_Simulator
from .exporters import PropsimASCIIExporter
from .data_models import (
    ChamberConfig, TargetChannelConfig, PropsimExportConfig, CalibrationConfig,
)

# B-1 烘焙允许的 target_path (确定性相位 / 烘焙路); B2_parametric 走 .tap 参数化路。
_B1_TARGET_PATHS = ('B1_baked', 'GCM_native')


def _snapshot_to_custom(annotated: AnnotatedCDLProfile, snap: CDLSnapshot) -> CustomCDLProfile:
    """AnnotatedCDLProfile 的一个快照 → CustomCDLProfile (标准几何, 喂 CustomCDLBuilder)。

    只搬标准几何/功率字段 (与 CDLClusterSpec 对齐); 真实子径 (subray_sum) 经
    cluster_subrays 旁路单独透传, 不进 CustomCDLProfile。
    """
    return CustomCDLProfile(
        center_frequency_hz=annotated.center_frequency_hz,
        pathloss_db=annotated.pathloss_db,
        is_los=annotated.is_los,
        k_factor_db=annotated.k_factor_db,
        ue_velocity_mps=snap.ue_velocity_mps,
        clusters=[
            CDLClusterSpec(
                delay_s=c.delay_s, power_linear=c.power_linear,
                aoa_deg=c.aoa_deg, aod_deg=c.aod_deg,
                zoa_deg=c.zoa_deg, zod_deg=c.zod_deg,
                as_aoa_deg=c.as_aoa_deg, as_aod_deg=c.as_aod_deg,
                as_zoa_deg=c.as_zoa_deg, as_zod_deg=c.as_zod_deg,
                xpr_db=c.xpr_db, initial_phases_rad=c.initial_phases_rad,
            )
            for c in snap.clusters
        ],
    )


def _extract_subrays(snap: CDLSnapshot) -> Dict[int, List]:
    """提取 subray_sum 簇的真实子径 (cluster_index → [SubRay]); baked 簇不进 → 走标准扇。"""
    return {
        i: c.subrays
        for i, c in enumerate(snap.clusters)
        if c.doppler_repr.kind == 'subray_sum' and c.subrays
    }


def _select_snapshot(annotated: AnnotatedCDLProfile, snapshot_index: int) -> CDLSnapshot:
    """按 declared CDLSnapshot.index 选快照 (非 list offset)。schema 仅保证 index 集合
    连续 0..n-1, 不保证 snapshots 列表顺序 == index 顺序 (Codex P2: 乱序 list 上用 offset
    会烘错帧)。index 在 profile 内唯一, 故至多匹配一个。"""
    matching = [s for s in annotated.snapshots if s.index == snapshot_index]
    if not matching:
        raise IndexError(
            f"snapshot_index={snapshot_index} 不在 profile 快照 index 中 "
            f"(可用: {sorted(s.index for s in annotated.snapshots)})"
        )
    return matching[0]


def bake_b1_annotated(
    annotated: AnnotatedCDLProfile,
    chamber_config: ChamberConfig,
    channel_config: TargetChannelConfig,
    export_config: PropsimExportConfig,
    calibration_config: Optional[CalibrationConfig] = None,
    *,
    synthesis_method: str = 'ray',
    output_path: Optional[str] = None,
    direction: str = "dl",
    snapshot_index: int = 0,
) -> List[str]:
    """AnnotatedCDLProfile (B1_baked / GCM_native) → per-(Tx, Probe) .asc (OTA, 含校准)。

    Args:
        annotated: 标注式 CDL profile; target_path 须为 B1_baked / GCM_native。
        chamber_config / channel_config: 暗室+探头 / DUT 天线配置 (正交于 CDL)。
        export_config / calibration_config: .asc 导出参数 / OTA 校准 (None=不施加)。
        synthesis_method: 'ray' (几何 GSCM, 默认) 或 'strict_pfs' (逐探头独立衰落)。
            subray_sum 簇用确定性 phase_rad, 不受此项影响; 仅 baked (标准扇) 簇受影响。
        snapshot_index: 多快照时烘第几个 (默认首快照)。

    Returns: 生成的 .asc 文件路径列表 (per Tx→Probe link)。
    """
    tp = annotated.clustering_meta.target_path
    if tp not in _B1_TARGET_PATHS:
        raise ValueError(
            f"bake_b1_annotated 仅接受 target_path={_B1_TARGET_PATHS} (B-1 确定性烘焙); "
            f"got {tp!r}。B2_parametric 走 B-2 .tap 参数化路 (硬件实时衰落), 非本烘焙器。"
        )

    n_snap = len(annotated.snapshots)
    snap = _select_snapshot(annotated, snapshot_index)
    if n_snap > 1:
        warnings.warn(
            f"AnnotatedCDLProfile 有 {n_snap} 个快照 (动态轨迹), 本烘焙器单快照先行, "
            f"仅烘 snapshot index={snapshot_index}; 跨快照 .asc 帧拼接进 backlog。",
            UserWarning, stacklevel=2,
        )
    custom = _snapshot_to_custom(annotated, snap)
    cluster_subrays = _extract_subrays(snap)

    profile = CustomCDLBuilder(custom).build()
    sim_result = MIMO_OTA_Simulator().synthesize_ota(
        chamber_config, channel_config, profile,
        synthesis_method=synthesis_method, cluster_subrays=cluster_subrays,
    )
    exporter = PropsimASCIIExporter(export_config, calibration_config)
    return exporter.export(sim_result, output_path=output_path, direction=direction)
