"""F3 — geometric_native_fit 聚类 (单快照) (P2-14 B-2 信道注入).

把 RT 射线 (MPC) 聚成 ≤ tap_budget 个簇, 使【每簇都 F64 原生谱可表示】, 输出
B-2 标注式 CDL (AnnotatedCDLProfile)。这是把"F64 无 custom PSD"的 blocker 在合成层
绕过去的核心 IP (V1.0 §4.1): 聚类是我们的自由度 + 降到 ≤24 抽头本就不可避免, 于是
按 (时延, 角度, 多普勒) 把簇切到每簇落在原生谱上。

算法 (单快照, 确定性):
  1. 按时延把 MPC 初聚成簇 (F64 抽头在不同时延)。
  2. 若初始簇数 > tap_budget, 按时延邻近 + 功率弱优先合并到 budget。
  3. native-fit 分裂循环: 对每个 needs_split 的簇 (F1 最优原生残差 > rho_thresh),
     按多普勒中位数二分; budget-aware (总簇数 < tap_budget 才分; 功率大者优先分)。
  4. 每簇经 F1 映射成 doppler_repr (native / dual_gaussian), 装配 AnnotatedCDLProfile。

依赖 F1 (doppler_native_mapping) + F2 (annotated_cdl_schema)。纯函数, 仅 numpy + pydantic。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .annotated_cdl_schema import (
    AnnotatedCDLProfile,
    AnnotatedCluster,
    CDLSnapshot,
    ClusteringMeta,
    DopplerDualGaussian,
    DopplerNative,
    SubRay,
)
from .doppler_native_mapping import _doppler_from_angles, map_doppler_to_native


@dataclass(frozen=True)
class MPC:
    """一条多径分量 (RT 射线)。"""
    delay_s: float
    power_linear: float
    aoa_deg: float
    aod_deg: float
    zoa_deg: float = 90.0
    zod_deg: float = 90.0


@dataclass(frozen=True)
class F64CapabilityProfile:
    """F64 能力档 (聚类约束); 现场标定见 V1.0 §9。"""
    name: str = 'default'
    tap_budget: int = 24            # 每逻辑通道最大抽头
    rho_thresh: float = 0.06        # native 拟合残差验收阈
    delay_resolution_s: float = 5e-9


@dataclass
class _Cluster:
    """中间簇 (聚类过程态)。"""
    mpcs: List[MPC]
    parent_id: Optional[int] = None

    @property
    def power(self) -> float:
        return sum(m.power_linear for m in self.mpcs)

    @property
    def delay_centroid(self) -> float:
        w = np.array([m.power_linear for m in self.mpcs])
        d = np.array([m.delay_s for m in self.mpcs])
        return float(np.sum(w * d) / w.sum())


# ───────────────────── 步骤 ─────────────────────

def _initial_delay_clusters(mpcs: Sequence[MPC], delay_res: float) -> List[_Cluster]:
    """按时延贪心初聚: 排序后, 与当前组首成员时延差 ≤ delay_res 的并入同组。"""
    if not mpcs:
        return []
    ordered = sorted(mpcs, key=lambda m: m.delay_s)
    groups: List[List[MPC]] = [[ordered[0]]]
    for m in ordered[1:]:
        if m.delay_s - groups[-1][0].delay_s <= delay_res:
            groups[-1].append(m)
        else:
            groups.append([m])
    return [_Cluster(mpcs=g) for g in groups]


def _merge_to_budget(clusters: List[_Cluster], tap_budget: int) -> List[_Cluster]:
    """初始簇数超 budget: 反复合并【时延最近的一对】直到 ≤ budget。"""
    clusters = list(clusters)
    while len(clusters) > tap_budget:
        clusters.sort(key=lambda c: c.delay_centroid)
        # 找相邻时延差最小的一对合并
        gaps = [clusters[i + 1].delay_centroid - clusters[i].delay_centroid
                for i in range(len(clusters) - 1)]
        i = int(np.argmin(gaps))
        merged = _Cluster(mpcs=clusters[i].mpcs + clusters[i + 1].mpcs)
        clusters[i:i + 2] = [merged]
    return clusters


def _cluster_native(cluster: _Cluster, velocity, f_c, rho_thresh: float):
    """对一个簇跑 F1 原生映射 (用簇内 MPC 角度 + 功率)。"""
    aoa = [m.aoa_deg for m in cluster.mpcs]
    zoa = [m.zoa_deg for m in cluster.mpcs]
    power = [m.power_linear for m in cluster.mpcs]
    p = np.array(power, dtype=float)
    fd = _doppler_from_angles(np.array(aoa), np.array(zoa), velocity, f_c)
    return map_doppler_to_native(fd, p / p.sum(), split_threshold=rho_thresh)


def _doppler_median_split(cluster: _Cluster, velocity, f_c) -> Optional[Tuple[_Cluster, _Cluster]]:
    """按【功率加权多普勒中位数】把簇二分。返回 (低频侧, 高频侧) 或 None (不可分)。"""
    if len(cluster.mpcs) < 2:
        return None
    fd = _doppler_from_angles(
        np.array([m.aoa_deg for m in cluster.mpcs]),
        np.array([m.zoa_deg for m in cluster.mpcs]), velocity, f_c)
    p = np.array([m.power_linear for m in cluster.mpcs])
    order = np.argsort(fd)
    cdf = np.cumsum(p[order]) / p.sum()
    cut = np.searchsorted(cdf, 0.5)
    cut = min(max(cut, 1), len(cluster.mpcs) - 1)        # 两侧都非空
    lo_idx = order[:cut]
    hi_idx = order[cut:]
    lo = _Cluster(mpcs=[cluster.mpcs[i] for i in lo_idx], parent_id=cluster.parent_id)
    hi = _Cluster(mpcs=[cluster.mpcs[i] for i in hi_idx], parent_id=cluster.parent_id)
    return lo, hi


def _native_fit_split(clusters: List[_Cluster], velocity, f_c,
                      profile: F64CapabilityProfile) -> List[_Cluster]:
    """budget-aware 分裂: 反复把功率最大的 needs_split 簇二分, 直到无 needs_split 或撞 budget。"""
    work = list(clusters)
    while len(work) < profile.tap_budget:
        # 找功率最大的 needs_split 且可分的簇
        candidates = []
        for idx, c in enumerate(work):
            res = _cluster_native(c, velocity, f_c, profile.rho_thresh)
            if res.needs_split and len(c.mpcs) >= 2:
                candidates.append((c.power, idx))
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, idx = candidates[0]
        split = _doppler_median_split(work[idx], velocity, f_c)
        if split is None:
            break
        # 分裂血缘: 子簇 parent = 原簇在最终列表里的序号 (装配时回填); 这里先标 origin idx
        lo, hi = split
        lo.parent_id = work[idx].parent_id if work[idx].parent_id is not None else idx
        hi.parent_id = lo.parent_id
        work[idx:idx + 1] = [lo, hi]
    return work


def _build_annotated(cluster: _Cluster, cluster_id: int, velocity, f_c,
                     rho_thresh: float) -> Tuple[AnnotatedCluster, float]:
    """把一个最终簇装配成 AnnotatedCluster (经 F1 定 doppler_repr + 携带子径)。"""
    res = _cluster_native(cluster, velocity, f_c, rho_thresh)
    p = np.array([m.power_linear for m in cluster.mpcs])
    w = p / p.sum()
    # 功率加权代表角度
    aoa = float(np.sum(w * np.array([m.aoa_deg for m in cluster.mpcs])))
    aod = float(np.sum(w * np.array([m.aod_deg for m in cluster.mpcs])))
    zoa = float(np.sum(w * np.array([m.zoa_deg for m in cluster.mpcs])))
    zod = float(np.sum(w * np.array([m.zod_deg for m in cluster.mpcs])))
    # 角扩展 (绕代表角的加权 std)
    as_aoa = float(np.sqrt(np.sum(w * (np.array([m.aoa_deg for m in cluster.mpcs]) - aoa) ** 2)))
    delay = float(np.sum(w * np.array([m.delay_s for m in cluster.mpcs])))

    if res.kind == 'dual_gaussian':
        doppler = DopplerDualGaussian(
            beam_a=res.params['beam_a'], beam_b=res.params['beam_b'],
            power_ratio_db=res.params['power_ratio_db'], native_fit_residual=res.residual)
    else:
        doppler = DopplerNative(
            shape=res.shape, f_d_centroid_hz=res.f_d_centroid_hz,
            f_d_max_hz=res.f_d_max_hz, shape_params=(res.params or None),
            native_fit_residual=res.residual)

    subrays = [SubRay(aoa_deg=m.aoa_deg, aod_deg=m.aod_deg, zoa_deg=m.zoa_deg,
                      zod_deg=m.zod_deg, power_linear=m.power_linear) for m in cluster.mpcs]
    ac = AnnotatedCluster(
        cluster_id=cluster_id,
        parent_cluster_id=cluster.parent_id,
        delay_s=delay, power_linear=float(p.sum()),
        aoa_deg=aoa, aod_deg=aod, zoa_deg=zoa, zod_deg=zod,
        as_aoa_deg=as_aoa,
        doppler_repr=doppler, physicality='rt_physical', subrays=subrays)
    return ac, res.residual


def geometric_native_fit(
    mpcs: Sequence[MPC], velocity_mps: Sequence[float], f_c_hz: float,
    profile: F64CapabilityProfile, *,
    test_class: str = 'throughput_psd',
    pathloss_db: float = 0.0, is_los: bool = False, k_factor_db: Optional[float] = None,
) -> AnnotatedCDLProfile:
    """RT 射线 → ≤tap_budget 个 native 可表示簇 → B-2 标注式 CDL (单快照)。"""
    if not mpcs:
        raise ValueError("mpcs 不能为空")

    clusters = _initial_delay_clusters(mpcs, profile.delay_resolution_s)
    clusters = _merge_to_budget(clusters, profile.tap_budget)
    clusters = _native_fit_split(clusters, velocity_mps, f_c_hz, profile)

    annotated: List[AnnotatedCluster] = []
    residuals: List[float] = []
    for cid, c in enumerate(clusters):
        ac, resid = _build_annotated(c, cid, velocity_mps, f_c_hz, profile.rho_thresh)
        annotated.append(ac)
        residuals.append(resid)

    return AnnotatedCDLProfile(
        center_frequency_hz=f_c_hz, pathloss_db=pathloss_db, is_los=is_los,
        k_factor_db=k_factor_db,
        clustering_meta=ClusteringMeta(
            algorithm='geometric_native_fit', target_path='B2_parametric',
            test_class=test_class, physicality='rt_physical',
            f64_profile_ref=profile.name, tap_budget_used=len(clusters),
            max_native_fit_residual=(max(residuals) if residuals else None)),
        snapshots=[CDLSnapshot(index=0, ue_velocity_mps=tuple(velocity_mps),
                               clusters=annotated)],
    )
