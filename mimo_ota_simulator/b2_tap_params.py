"""PR-4 — B-2 per-tap 参数表提取: AnnotatedCDLProfile(native/dual_gaussian) → 参数表 (P2-14)。

把 B-2 标注式 CDL 的每个 native/dual_gaussian 簇映射成一个 F64 .tap per-tap 参数行
{τ, P, 原生谱形, 质心, 展宽, 角度, K}。这一步把 B-2 生成缺口从"整个生成层"收缩到
【仅 .tap 字节序列化】—— 字节字段顺序 / gaussian 谱关键字 / f_upd 等现场标定 (V1.0 §9.1),
本模块只到结构化参数层 (可独立检视 + 单测)。

target_path 须为 B2_parametric (native/dual_gaussian); B-1/GCM (baked/subray_sum) 走烘焙路。
单快照先行; 多快照 → .rtc 多 environment (per-snapshot tap 表), 进 backlog。

纯函数, 无硬件 / 无 I/O。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

from .annotated_cdl_schema import AnnotatedCDLProfile, CDLSnapshot


@dataclass(frozen=True)
class TapParameter:
    """一个 F64 .tap per-tap 参数行 (B-2 硬件实时衰落)。

    doppler_spread_hz: native=f_d_max_hz (绕质心展宽半宽); dual_gaussian=max(两 beam std)。
    doppler_params: native=shape_params (如 gaussian:{std_hz}/rice:{k_db}); dual_gaussian=
                    {beam_a, beam_b, power_ratio_db}。
    """
    tap_index: int
    delay_s: float
    power_linear: float
    # 多普勒谱 (F64 原生)
    doppler_kind: str               # 'native' | 'dual_gaussian'
    doppler_shape: str              # native: classical/gaussian/rounded/flat/pure_doppler/rice; dual_gaussian: 'gaussian'
    doppler_centroid_hz: float
    doppler_spread_hz: float
    doppler_params: dict = field(default_factory=dict)
    # 角度 (下游探头 PAS 映射消费)
    aoa_deg: float = 0.0
    aod_deg: float = 0.0
    zoa_deg: float = 90.0
    zod_deg: float = 90.0
    as_aoa_deg: float = 0.0
    as_aod_deg: float = 0.0
    as_zoa_deg: float = 0.0     # 仰角展宽 (3D RT 仰角簇 + 多环布局 PAS 映射需要, Codex P2)
    as_zod_deg: float = 0.0
    # 极化 / 拟合质量
    xpr_db: Optional[float] = None
    native_fit_residual: float = 0.0


def _select_snapshot(annotated: AnnotatedCDLProfile, snapshot_index: int) -> CDLSnapshot:
    """按 declared CDLSnapshot.index 选 (非 list offset; schema 仅保证 index 集合连续)。"""
    matching = [s for s in annotated.snapshots if s.index == snapshot_index]
    if not matching:
        raise IndexError(
            f"snapshot_index={snapshot_index} 不在 profile 快照 index 中 "
            f"(可用: {sorted(s.index for s in annotated.snapshots)})"
        )
    return matching[0]


def _dual_gaussian_centroid_hz(beam_a, beam_b, power_ratio_db: float) -> float:
    """双高斯整体质心 = 功率加权 beam center (power_ratio_db = P(beam_a)/P(beam_b))。"""
    p_a = 10.0 ** (power_ratio_db / 10.0)
    p_b = 1.0
    return (p_a * beam_a.center_shift_hz + p_b * beam_b.center_shift_hz) / (p_a + p_b)


def extract_tap_parameters(
    annotated: AnnotatedCDLProfile,
    snapshot_index: int = 0,
) -> List[TapParameter]:
    """AnnotatedCDLProfile (B2_parametric) → per-tap 参数表 (单快照)。

    Raises ValueError if target_path != B2_parametric 或某簇非 native/dual_gaussian。
    """
    tp = annotated.clustering_meta.target_path
    if tp != 'B2_parametric':
        raise ValueError(
            f"extract_tap_parameters 仅接受 target_path=B2_parametric (B-2 参数化路); "
            f"got {tp!r}。B1_baked/GCM_native (baked/subray_sum) 走 b1_annotated_baker 烘焙路。"
        )
    snap = _select_snapshot(annotated, snapshot_index)
    if len(annotated.snapshots) > 1:
        warnings.warn(
            f"AnnotatedCDLProfile 有 {len(annotated.snapshots)} 个快照 (动态轨迹), 本提取器单"
            f"快照先行, 仅取 snapshot index={snapshot_index}; 多快照 .rtc 多 environment 进 backlog。",
            UserWarning, stacklevel=2,
        )

    taps: List[TapParameter] = []
    for i, c in enumerate(snap.clusters):
        dr = c.doppler_repr
        if dr.kind == 'native':
            taps.append(TapParameter(
                tap_index=i, delay_s=c.delay_s, power_linear=c.power_linear,
                doppler_kind='native', doppler_shape=dr.shape,
                doppler_centroid_hz=dr.f_d_centroid_hz, doppler_spread_hz=dr.f_d_max_hz,
                doppler_params=dict(dr.shape_params or {}),
                aoa_deg=c.aoa_deg, aod_deg=c.aod_deg, zoa_deg=c.zoa_deg, zod_deg=c.zod_deg,
                as_aoa_deg=c.as_aoa_deg, as_aod_deg=c.as_aod_deg,
                as_zoa_deg=c.as_zoa_deg, as_zod_deg=c.as_zod_deg,
                xpr_db=c.xpr_db, native_fit_residual=dr.native_fit_residual))
        elif dr.kind == 'dual_gaussian':
            taps.append(TapParameter(
                tap_index=i, delay_s=c.delay_s, power_linear=c.power_linear,
                doppler_kind='dual_gaussian', doppler_shape='gaussian',
                doppler_centroid_hz=_dual_gaussian_centroid_hz(dr.beam_a, dr.beam_b, dr.power_ratio_db),
                doppler_spread_hz=max(dr.beam_a.std_hz, dr.beam_b.std_hz),
                doppler_params={
                    'beam_a': {'center_shift_hz': dr.beam_a.center_shift_hz, 'std_hz': dr.beam_a.std_hz},
                    'beam_b': {'center_shift_hz': dr.beam_b.center_shift_hz, 'std_hz': dr.beam_b.std_hz},
                    'power_ratio_db': dr.power_ratio_db,
                },
                aoa_deg=c.aoa_deg, aod_deg=c.aod_deg, zoa_deg=c.zoa_deg, zod_deg=c.zod_deg,
                as_aoa_deg=c.as_aoa_deg, as_aod_deg=c.as_aod_deg,
                as_zoa_deg=c.as_zoa_deg, as_zod_deg=c.as_zod_deg,
                xpr_db=c.xpr_db, native_fit_residual=dr.native_fit_residual))
        else:
            # baked / subray_sum 是 B-1/GCM 烘焙路, 不该出现在 B2_parametric profile
            # (schema §5.4 已挡 target↔kind 自洽, 这里再防御一层)。
            raise ValueError(
                f"B2_parametric 簇 {i} 的 doppler_repr.kind={dr.kind!r} 非 native/dual_gaussian; "
                f"baked/subray_sum 走 B-1 烘焙路 (b1_annotated_baker)"
            )
    return taps
