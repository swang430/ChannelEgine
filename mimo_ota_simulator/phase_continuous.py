"""F5 — phase_continuous 聚类 (单快照) (P2-14 B-2 信道注入).

确定性相位测试 (ISAC / 波束跟踪) 走 B-1/GCM, 用本聚类: 把 RT 射线聚成 ≤ tap_budget
个簇, 【不要求 native 可表示】(B-1 烘焙 / GCM 求和任意形状), 但【保留子径确定性初相
phase_rad】, doppler_repr=subray_sum, 供下游确定性子径相干求和。

跟 geometric_native_fit (F3, B-2 统计路) 的区别:
  - 不做 native-fit 分裂 (烘什么形状都行) —— 只按时延聚类 + 超 budget 合并;
  - doppler_repr = subray_sum (不是 native/dual_gaussian);
  - 子径【必须】带 phase_rad (F2 §5.4 校验);
  - target_path = B1_baked / GCM_native, algorithm = phase_continuous。

复用 F3 的时延聚类 helper。依赖 F3 + F2。纯函数, 仅 numpy + pydantic。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .annotated_cdl_schema import (
    AnnotatedCDLProfile,
    AnnotatedCluster,
    CDLSnapshot,
    ClusteringMeta,
    DopplerSubraySum,
    SubRay,
)
from .doppler_native_mapping import _doppler_from_angles
from .geometric_native_fit import (
    MPC,
    F64CapabilityProfile,
    _Cluster,
    _initial_delay_clusters,
    _merge_to_budget,
    _wmean_spread,
)


def _build_subray_sum_cluster(cluster: _Cluster, cluster_id: int, velocity, f_c) -> AnnotatedCluster:
    """装配成 subray_sum 簇 (确定性相位; 携带子径 + phase_rad)。"""
    w = np.array([m.power_linear for m in cluster.mpcs], dtype=float)
    aoa, as_aoa = _wmean_spread([m.aoa_deg for m in cluster.mpcs], w, circular=True)
    aod, as_aod = _wmean_spread([m.aod_deg for m in cluster.mpcs], w, circular=True)
    zoa, as_zoa = _wmean_spread([m.zoa_deg for m in cluster.mpcs], w, circular=False)
    zod, as_zod = _wmean_spread([m.zod_deg for m in cluster.mpcs], w, circular=False)
    delay, _ = _wmean_spread([m.delay_s for m in cluster.mpcs], w, circular=False)

    # 簇质心多普勒 (informational; 大质心可拆出走硬件频偏)
    fd = _doppler_from_angles(
        np.array([m.aoa_deg for m in cluster.mpcs]),
        np.array([m.zoa_deg for m in cluster.mpcs]), velocity, f_c)
    centroid = float(np.sum((w / w.sum()) * fd))

    subrays = [SubRay(aoa_deg=m.aoa_deg, aod_deg=m.aod_deg, zoa_deg=m.zoa_deg,
                      zod_deg=m.zod_deg, power_linear=m.power_linear, phase_rad=m.phase_rad)
               for m in cluster.mpcs]
    return AnnotatedCluster(
        cluster_id=cluster_id, delay_s=delay, power_linear=float(w.sum()),
        aoa_deg=aoa, aod_deg=aod, zoa_deg=zoa, zod_deg=zod,
        as_aoa_deg=as_aoa, as_aod_deg=as_aod, as_zoa_deg=as_zoa, as_zod_deg=as_zod,
        doppler_repr=DopplerSubraySum(f_d_centroid_hz=centroid),
        physicality='rt_physical', subrays=subrays)


def phase_continuous_fit(
    mpcs: Sequence[MPC], velocity_mps: Sequence[float], f_c_hz: float,
    profile: F64CapabilityProfile, *,
    target_path: str = 'B1_baked',
    test_class: str = 'isac_sensing',
    pathloss_db: float = 0.0, is_los: bool = False, k_factor_db: Optional[float] = None,
) -> AnnotatedCDLProfile:
    """RT 射线 → ≤tap_budget 个确定性相位簇 → B-1/GCM 标注式 CDL (单快照)。"""
    if not mpcs:
        raise ValueError("mpcs 不能为空")
    if target_path not in ('B1_baked', 'GCM_native'):
        raise ValueError(f"phase_continuous 仅支持 target_path B1_baked / GCM_native, got {target_path}")
    if any(m.phase_rad is None for m in mpcs):
        raise ValueError("phase_continuous 须每条 MPC 带确定性 phase_rad (subray_sum 要求)")

    clusters = _initial_delay_clusters(mpcs, profile.delay_resolution_s)
    clusters = _merge_to_budget(clusters, profile.tap_budget)   # 不做 native-fit 分裂

    annotated = [_build_subray_sum_cluster(c, cid, velocity_mps, f_c_hz)
                 for cid, c in enumerate(clusters)]

    return AnnotatedCDLProfile(
        center_frequency_hz=f_c_hz, pathloss_db=pathloss_db, is_los=is_los, k_factor_db=k_factor_db,
        clustering_meta=ClusteringMeta(
            algorithm='phase_continuous', target_path=target_path,
            test_class=test_class, physicality='rt_physical',
            f64_profile_ref=profile.name, tap_budget_used=len(clusters)),
        snapshots=[CDLSnapshot(index=0, ue_velocity_mps=tuple(velocity_mps),
                               clusters=annotated)],
    )
