"""F4 — 时空跟踪层 (P2-14 B-2 信道注入).

把【一条轨迹的多快照 RT 射线】聚成一个时间演化的 B-2 标注式 CDL: 每快照跑 F3
(geometric_native_fit), 再【跨快照跟踪】同一物理簇 —— 维持 cluster_id 一致、平滑演化、
生灭 (birth/death) 标注。这是 V1.0 §4.3 的连续性命根: 逐快照独立聚类会让簇身份跳变,
毁掉 RT 的相位/空间连续性 + F64 快照间插值。

跟踪: 相邻快照间按 (时延, 方位角) 贪心一对一匹配 (双门限); 匹配上 → 继承全局 id;
未匹配的当前簇 → 新生 (birth); 上一快照未被匹配的簇 → 消亡 (death)。

依赖 F3 (geometric_native_fit) + F2 (annotated_cdl_schema)。纯函数, 仅 numpy + pydantic。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .annotated_cdl_schema import (
    AnnotatedCDLProfile,
    AnnotatedCluster,
    CDLSnapshot,
    ClusteringMeta,
    TrajectoryMeta,
)
from .geometric_native_fit import F64CapabilityProfile, MPC, geometric_native_fit


def _ang_diff_deg(a: float, b: float) -> float:
    """圆周角度差 (deg), ∈ [0,180]。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _normalize_velocities(velocities, n: int) -> List[tuple]:
    """单个 3-vec → 全快照共享; nx3 → per-snapshot。"""
    arr = np.asarray(velocities, dtype=float)
    if arr.shape == (3,):
        return [tuple(float(x) for x in arr)] * n
    if arr.shape == (n, 3):
        return [tuple(float(x) for x in v) for v in arr]
    raise ValueError(f"velocities 须是单个 3-vec 或 {n}x3, got shape {arr.shape}")


def _match_greedy(prev: Sequence[AnnotatedCluster], curr: Sequence[AnnotatedCluster],
                  delay_gate_s: float, angle_gate_deg: float) -> Dict[int, int]:
    """相邻快照贪心一对一匹配 → {curr.cluster_id: prev.cluster_id}。双门限内按归一化距离。"""
    pairs = []
    for ci in curr:
        for pj in prev:
            dd = abs(ci.delay_s - pj.delay_s)
            da = _ang_diff_deg(ci.aoa_deg, pj.aoa_deg)
            if dd <= delay_gate_s and da <= angle_gate_deg:
                pairs.append((dd / delay_gate_s + da / angle_gate_deg, ci.cluster_id, pj.cluster_id))
    pairs.sort()
    used_c, used_p, match = set(), set(), {}
    for _, cid, pid in pairs:
        if cid in used_c or pid in used_p:
            continue
        match[cid] = pid
        used_c.add(cid)
        used_p.add(pid)
    return match


def native_fit_trajectory(
    snapshots_mpcs: Sequence[Sequence[MPC]],
    velocities,
    f_c_hz: float,
    profile: F64CapabilityProfile,
    *,
    skeleton_rate_hz: float,
    delta_d_geo_m: float,
    route_len_m: float,
    test_class: str = 'throughput_psd',
    pathloss_db: float = 0.0,
    is_los: bool = False,
    k_factor_db: Optional[float] = None,
    delay_gate_s: float = 1e-8,
    angle_gate_deg: float = 20.0,
) -> AnnotatedCDLProfile:
    """多快照 RT 射线 → 跨快照跟踪的 B-2 标注式 CDL (动态轨迹)。"""
    if not snapshots_mpcs:
        raise ValueError("snapshots_mpcs 不能为空")
    n_snap = len(snapshots_mpcs)
    vels = _normalize_velocities(velocities, n_snap)

    # 1. 每快照跑 F3 (得 local cluster_id 的簇)
    per_snap: List[List[AnnotatedCluster]] = []
    for mpcs, v in zip(snapshots_mpcs, vels):
        prof = geometric_native_fit(mpcs, v, f_c_hz, profile, test_class=test_class)
        per_snap.append(list(prof.snapshots[0].clusters))

    # 2. 跨快照跟踪 → 每快照 local_id -> global_id
    global_maps: List[Dict[int, int]] = []
    next_global = 0
    for s_idx, clusters in enumerate(per_snap):
        l2g: Dict[int, int] = {}
        if s_idx == 0:
            for c in clusters:
                l2g[c.cluster_id] = next_global
                next_global += 1
        else:
            match = _match_greedy(per_snap[s_idx - 1], clusters, delay_gate_s, angle_gate_deg)
            prev_l2g = global_maps[s_idx - 1]
            for c in clusters:
                if c.cluster_id in match:
                    l2g[c.cluster_id] = prev_l2g[match[c.cluster_id]]   # 继承
                else:
                    l2g[c.cluster_id] = next_global                     # 新生
                    next_global += 1
        global_maps.append(l2g)

    # 3. 每个 global id 出现的快照 (生命周期)
    appears: Dict[int, List[int]] = defaultdict(list)
    for s_idx, l2g in enumerate(global_maps):
        for g in l2g.values():
            appears[g].append(s_idx)

    # 4. 重组多快照 profile (重映射 id + parent + 生灭)
    out_snaps: List[CDLSnapshot] = []
    for s_idx, (clusters, l2g, v) in enumerate(zip(per_snap, global_maps, vels)):
        out_clusters = []
        for c in clusters:
            g = l2g[c.cluster_id]
            life = appears[g]
            first, last = life[0], life[-1]
            if s_idx == first and first > 0:
                ramp = 'birth'                      # 轨迹中途新生
            elif s_idx == last and last < n_snap - 1:
                ramp = 'death'                      # 轨迹中途消亡
            else:
                ramp = 'stable'
            parent_g = l2g[c.parent_cluster_id] if c.parent_cluster_id is not None else None
            out_clusters.append(c.model_copy(update={
                'cluster_id': g, 'parent_cluster_id': parent_g,
                'birth_index': first, 'death_index': last, 'power_ramp': ramp}))
        out_snaps.append(CDLSnapshot(index=s_idx, ue_velocity_mps=tuple(v), clusters=out_clusters))

    return AnnotatedCDLProfile(
        center_frequency_hz=f_c_hz, pathloss_db=pathloss_db, is_los=is_los, k_factor_db=k_factor_db,
        clustering_meta=ClusteringMeta(
            algorithm='geometric_native_fit', target_path='B2_parametric',
            test_class=test_class, physicality='rt_physical', f64_profile_ref=profile.name,
            tap_budget_used=max(len(c) for c in per_snap)),
        trajectory_meta=TrajectoryMeta(
            n_snapshots=n_snap, skeleton_rate_hz=skeleton_rate_hz,
            delta_d_geo_m=delta_d_geo_m, route_len_m=route_len_m),
        snapshots=out_snaps,
    )
