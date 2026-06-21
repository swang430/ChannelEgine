"""PR-3 — §6 路径判决: test_class 驱动选 B-1 / B-2 / GCM (P2-14)。

设计 V1.0 §6 (突破后大幅简化):
  - throughput_psd / consistency (统计衰落可接受): B-2 普及 (native-fit 可表示则 B-2,
    否则 B-1 退路)。
  - isac_sensing / beam_tracking (需确定性相位 ↔ 几何可反演): B-1 (低 f_D,max) 或 GCM
    (大质心 / FR2 高速, B-1 奈奎斯特烘不动)。GCM 必需但无 license → ESCALATE fail-loud,
    【绝不静默回落 B1_baked】(否则选到不可行路径; 同 Codex P2 #165 母题)。

f_D,max / 文件经济从 V0.1 主边界降级为 B-1 被选中后的可行性校验 (附录 A); 这里用
f_D,max = |v|·f_c/c (与角度无关的上界, 保守) 判大质心。判决阈值见 F64CapabilityProfile
(现场标定 §9.7)。

纯函数, 无硬件 / 无 I/O。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import numpy as np

from .geometric_native_fit import MPC, F64CapabilityProfile, geometric_native_fit

_C = 3e8

TestClass = Literal['throughput_psd', 'consistency', 'isac_sensing', 'beam_tracking']
TargetPath = Literal['B2_parametric', 'B1_baked', 'GCM_native', 'ESCALATE']
ClusteringAlgo = Literal['geometric_native_fit', 'phase_continuous']

_DETERMINISTIC_PHASE_CLASSES = ('isac_sensing', 'beam_tracking')


@dataclass(frozen=True)
class PathDecision:
    """§6 判决结果。target_path='ESCALATE' = 需 GCM 但无 license, 调用方须 fail-loud,
    不得静默回落到任何可生成路径。"""
    target_path: TargetPath
    clustering_algo: ClusteringAlgo
    reason: str
    f_d_max_hz: float
    native_fit_residual: Optional[float] = None   # B-2 路实测 native 残差 (判决时已算)

    @property
    def is_escalation(self) -> bool:
        return self.target_path == 'ESCALATE'


def f_d_max_hz(velocity_mps: Sequence[float], f_c_hz: float) -> float:
    """场景最大多普勒频移上界 f_D,max = |v|·f_c/c (与角度无关; 任一射线 |f_D| ≤ 此)。"""
    v = float(np.linalg.norm(np.asarray(velocity_mps, dtype=float)))
    return v * f_c_hz / _C


def _is_fr2_high_speed(f_c_hz: float, velocity_mps: Sequence[float],
                       f64: F64CapabilityProfile) -> bool:
    """FR2 高速区: f_c ≥ FR2 起始 且 |v| > 高速门 (亚毫米地图 + 奈奎斯特, B-1 不可行)。"""
    v = float(np.linalg.norm(np.asarray(velocity_mps, dtype=float)))
    return f_c_hz >= f64.fr2_freq_thresh_hz and v > f64.fr2_high_speed_mps


def select_path_and_clustering(
    test_class: TestClass,
    mpcs: List[MPC],
    velocity_mps: Sequence[float],
    f_c_hz: float,
    f64_profile: Optional[F64CapabilityProfile] = None,
) -> PathDecision:
    """按 test_class + 场景 + F64 能力档判决注入路径与聚类算法 (设计 §6)。

    Returns PathDecision; target_path='ESCALATE' 时调用方须 fail-loud (绝不静默回落)。
    """
    if not mpcs:
        raise ValueError("select_path_and_clustering: mpcs 为空, 无法判决")
    f64 = f64_profile or F64CapabilityProfile()
    fdmax = f_d_max_hz(velocity_mps, f_c_hz)

    # ── 确定性相位类 (ISAC / 波束跟踪): B-1 或 GCM, B-2 排除 (统计实现反演不出几何) ──
    if test_class in _DETERMINISTIC_PHASE_CLASSES:
        large_centroid = fdmax > f64.large_centroid_thresh_hz
        fr2_hs = _is_fr2_high_speed(f_c_hz, velocity_mps, f64)
        if large_centroid or fr2_hs:
            # 大质心 / FR2 高速 → B-1 奈奎斯特烘不动, 必须 GCM
            why = ('FR2 高速' if fr2_hs
                   else f'大质心 f_D,max={fdmax:.0f}Hz>{f64.large_centroid_thresh_hz:.0f}')
            if f64.has_gcm:
                return PathDecision(
                    'GCM_native', 'phase_continuous',
                    f"确定性相位 + {why} → B-1 不可行, GCM", fdmax)
            return PathDecision(
                'ESCALATE', 'phase_continuous',
                f"确定性相位需 GCM ({why}) 但 F64 无 GCM license; "
                f"绝不静默回落 B1_baked (会选到不可行路径)", fdmax)
        # 低 f_D,max 确定性 → B-1 可行 (运行时频偏承载小质心)
        return PathDecision(
            'B1_baked', 'phase_continuous',
            f"确定性相位 + 低 f_D,max ({fdmax:.0f}Hz ≤ {f64.large_centroid_thresh_hz:.0f}): "
            f"B-1 可行", fdmax)

    # ── 吞吐 / 一致性类: B-2 普及 (native-fit 可表示则 B-2) ──
    fit = geometric_native_fit(mpcs, velocity_mps, f_c_hz, f64)
    resid = fit.clustering_meta.max_native_fit_residual
    budget_used = fit.clustering_meta.tap_budget_used
    native_ok = (resid is None or resid <= f64.rho_thresh) and budget_used <= f64.tap_budget
    if native_ok:
        return PathDecision(
            'B2_parametric', 'geometric_native_fit',
            f"吞吐/一致性 + native 可表示 (残差 {resid}, 抽头 {budget_used}/{f64.tap_budget})",
            fdmax, native_fit_residual=resid)
    # native-vs-连续冲突区 → B-1 退路 (本轮 Q2)
    return PathDecision(
        'B1_baked', 'phase_continuous',
        f"吞吐/一致性但 native 不可表示 (残差 {resid} > {f64.rho_thresh} 或抽头 "
        f"{budget_used} > {f64.tap_budget}): B-1 退路", fdmax, native_fit_residual=resid)
