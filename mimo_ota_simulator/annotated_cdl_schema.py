"""F2 — 标注式 CDL schema (P2-14 B-2 信道注入).

MPDB 多聚类后输出的 CDL: 三层结构 (profile / snapshot / cluster) + `doppler_repr`
区分联合体, **标注每簇该如何被下游 (ChannelEgine 合成 / F64 驱动) 处理**, 据此确定性
分流到 B-1 / B-2 / GCM。

设计依据: 姊妹仓 MIMO-First `docs/design/RT-MPDB-CDL-F64-channel-injection-design_V1.0.md` §5。

向后兼容: 既有 `CustomCDLProfile` (cdl_schema.py) 可经 `AnnotatedCDLProfile.from_custom_profile`
转成单快照 + baked 标注的 annotated profile (= 现行 B-1 路径)。

纯 Pydantic v2, 无硬件 / 无 I/O。
"""
from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cdl_schema import CustomCDLProfile


# ───────────────────── DopplerRepr 区分联合体 (V1.0 §5.3) ─────────────────────

class DopplerNative(BaseModel):
    """B-2: 映射到 F64 原生谱 (Classical/Gaussian/Rounded/Flat/Pure Doppler/Rice)。"""
    model_config = ConfigDict(extra='forbid')
    kind: Literal['native'] = 'native'
    shape: Literal['classical', 'gaussian', 'rounded', 'flat', 'pure_doppler', 'rice']
    f_d_centroid_hz: float = 0.0
    f_d_max_hz: float = Field(0.0, ge=0.0)          # 绕质心展宽半宽
    shape_params: Optional[dict] = None             # gaussian:{std_hz}; rice:{k_db}; ...
    native_fit_residual: float = Field(0.0, ge=0.0)


class DopplerDualGaussian(BaseModel):
    """B-2: 双高斯逼近 (F64 Gaussian 双 beam)。"""
    model_config = ConfigDict(extra='forbid')
    kind: Literal['dual_gaussian'] = 'dual_gaussian'
    beam_a: dict                                    # {center_shift_hz, std_hz}
    beam_b: dict
    power_ratio_db: float = 0.0
    native_fit_residual: float = Field(0.0, ge=0.0)


class DopplerBaked(BaseModel):
    """B-1: 多普勒烘进 CIR (形状不参数化, 靠 subrays/CIR 序列承载)。"""
    model_config = ConfigDict(extra='forbid')
    kind: Literal['baked'] = 'baked'
    f_d_centroid_hz: float = 0.0                    # 大质心若拆出走硬件频偏
    cir_ref: Optional[str] = None                   # 指向 .asc/.mat/.ir (可后填)


class DopplerSubraySum(BaseModel):
    """B-1/GCM: 确定性子径相干求和 (靠 subrays[].phase_rad)。"""
    model_config = ConfigDict(extra='forbid')
    kind: Literal['subray_sum'] = 'subray_sum'
    deterministic_phase: Literal[True] = True
    f_d_centroid_hz: float = 0.0


DopplerRepr = Annotated[
    Union[DopplerNative, DopplerDualGaussian, DopplerBaked, DopplerSubraySum],
    Field(discriminator='kind'),
]

# target_path 允许的 doppler_repr.kind (§5.4 一致性)
_PATH_ALLOWED_KINDS = {
    'B2_parametric': {'native', 'dual_gaussian'},
    'B1_baked': {'baked', 'subray_sum'},
    'GCM_native': {'baked', 'subray_sum'},
}


# ───────────────────── 簇 / 子径 ─────────────────────

class SubRay(BaseModel):
    model_config = ConfigDict(extra='forbid')
    aoa_deg: float
    aod_deg: float
    zoa_deg: float = 90.0
    zod_deg: float = 90.0
    power_linear: float = Field(..., gt=0.0)
    phase_rad: Optional[float] = None               # 确定性初相 (subray_sum 必需)


class AnnotatedCluster(BaseModel):
    model_config = ConfigDict(extra='forbid')

    # 身份与跟踪 (连续性核心)
    cluster_id: int
    parent_cluster_id: Optional[int] = None         # 由分裂而来则指父簇
    birth_index: Optional[int] = None
    death_index: Optional[int] = None
    power_ramp: Literal['stable', 'birth', 'death'] = 'stable'

    # 标准几何 / 功率 (与 CDLClusterSpec 对齐)
    delay_s: float = Field(..., ge=0.0)
    power_linear: float = Field(..., gt=0.0)
    aoa_deg: float
    aod_deg: float
    zoa_deg: float = 90.0
    zod_deg: float = 90.0
    as_aoa_deg: float = Field(0.0, ge=0.0)
    as_aod_deg: float = Field(0.0, ge=0.0)
    as_zoa_deg: float = Field(0.0, ge=0.0)
    as_zod_deg: float = Field(0.0, ge=0.0)
    xpr_db: Optional[float] = None
    initial_phases_rad: Optional[List[float]] = Field(None, min_length=4, max_length=4)

    # 多普勒表示标注
    doppler_repr: DopplerRepr

    physicality: Literal['rt_physical', 'standard_fan'] = 'rt_physical'
    subrays: Optional[List[SubRay]] = None


# ───────────────────── 元数据 / 快照 ─────────────────────

class ClusteringMeta(BaseModel):
    model_config = ConfigDict(extra='forbid')
    algorithm: Literal['geometric_native_fit', 'phase_continuous', 'raw_passthrough']
    target_path: Literal['B2_parametric', 'B1_baked', 'GCM_native']
    test_class: Literal['throughput_psd', 'consistency', 'isac_sensing', 'beam_tracking']
    physicality: Literal['rt_physical', 'standard_fan', 'mixed'] = 'rt_physical'
    f64_profile_ref: str
    tap_budget_used: int = Field(..., ge=0)
    max_native_fit_residual: Optional[float] = None
    notes: Optional[str] = None


class TrajectoryMeta(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_snapshots: int = Field(..., ge=1)
    skeleton_rate_hz: float = Field(..., gt=0.0)
    delta_d_geo_m: float = Field(..., gt=0.0)
    route_len_m: float = Field(..., ge=0.0)
    cir_update_rate_hz: Optional[float] = None      # B-1 的 f_upd; B-2 留空


class CDLSnapshot(BaseModel):
    model_config = ConfigDict(extra='forbid')
    index: int = Field(..., ge=0)
    time_s: Optional[float] = None
    position_m: Optional[Tuple[float, float, float]] = None
    ue_velocity_mps: Tuple[float, float, float]
    clusters: List[AnnotatedCluster] = Field(..., min_length=1)


# ───────────────────── 顶层 profile ─────────────────────

class AnnotatedCDLProfile(BaseModel):
    model_config = ConfigDict(extra='forbid')

    center_frequency_hz: float = Field(..., gt=0.0)
    pathloss_db: float
    is_los: bool
    k_factor_db: Optional[float] = None

    clustering_meta: ClusteringMeta
    trajectory_meta: Optional[TrajectoryMeta] = None    # None = 单快照 (静态)
    snapshots: List[CDLSnapshot] = Field(..., min_length=1)

    # ---- §5.4 一致性校验 (fail-loud) ----
    @model_validator(mode='after')
    def _validate_consistency(self):
        tp = self.clustering_meta.target_path
        algo = self.clustering_meta.algorithm
        allowed = _PATH_ALLOWED_KINDS[tp]

        # phase_continuous 聚类不能配 B2 (确定性相位不能交统计衰落)
        if algo == 'phase_continuous' and tp == 'B2_parametric':
            raise ValueError(
                "phase_continuous 聚类不能配 target_path=B2_parametric (确定性相位不能交统计衰落)"
            )

        for snap in self.snapshots:
            seen_ids = set()
            for c in snap.clusters:
                # cluster_id 在单快照内唯一 (跨快照连续性是 F4 的事)
                if c.cluster_id in seen_ids:
                    raise ValueError(
                        f"snapshot {snap.index} 内 cluster_id={c.cluster_id} 重复"
                    )
                seen_ids.add(c.cluster_id)
                # target_path ↔ doppler_repr.kind 自洽
                if c.doppler_repr.kind not in allowed:
                    raise ValueError(
                        f"target_path={tp} 不允许 doppler_repr.kind={c.doppler_repr.kind} "
                        f"(snapshot {snap.index} cluster {c.cluster_id}); 允许 {sorted(allowed)}"
                    )
                # subray_sum 须每子径有确定性相位
                if c.doppler_repr.kind == 'subray_sum':
                    if not c.subrays or any(s.phase_rad is None for s in c.subrays):
                        raise ValueError(
                            f"subray_sum cluster {c.cluster_id} (snapshot {snap.index}) "
                            "须携带每子径 phase_rad"
                        )

        # 动态: trajectory_meta.n_snapshots 与实际快照数一致
        if self.trajectory_meta is not None:
            if self.trajectory_meta.n_snapshots != len(self.snapshots):
                raise ValueError(
                    f"trajectory_meta.n_snapshots={self.trajectory_meta.n_snapshots} "
                    f"≠ 实际快照数 {len(self.snapshots)}"
                )
        return self

    # ---- 向后兼容: CustomCDLProfile → 单快照 annotated (B-1 baked 默认标注) ----
    @classmethod
    def from_custom_profile(
        cls,
        profile: CustomCDLProfile,
        *,
        f64_profile_ref: str = 'legacy',
        test_class: Literal['throughput_psd', 'consistency', 'isac_sensing', 'beam_tracking'] = 'throughput_psd',
    ) -> 'AnnotatedCDLProfile':
        """既有 CustomCDLProfile → 单快照 AnnotatedCDLProfile。

        既有路径内部用标准 38.901 扇 + 烘焙, 故标注为 raw_passthrough / B1_baked /
        standard_fan, 每簇 doppler_repr=baked。语义等价于现行 B-1 行为。
        """
        clusters = [
            AnnotatedCluster(
                cluster_id=i,
                delay_s=c.delay_s,
                power_linear=c.power_linear,
                aoa_deg=c.aoa_deg, aod_deg=c.aod_deg,
                zoa_deg=c.zoa_deg, zod_deg=c.zod_deg,
                as_aoa_deg=c.as_aoa_deg, as_aod_deg=c.as_aod_deg,
                as_zoa_deg=c.as_zoa_deg, as_zod_deg=c.as_zod_deg,
                xpr_db=c.xpr_db,
                initial_phases_rad=c.initial_phases_rad,
                doppler_repr=DopplerBaked(),
                physicality='standard_fan',
            )
            for i, c in enumerate(profile.clusters)
        ]
        return cls(
            center_frequency_hz=profile.center_frequency_hz,
            pathloss_db=profile.pathloss_db,
            is_los=profile.is_los,
            k_factor_db=profile.k_factor_db,
            clustering_meta=ClusteringMeta(
                algorithm='raw_passthrough',
                target_path='B1_baked',
                test_class=test_class,
                physicality='standard_fan',
                f64_profile_ref=f64_profile_ref,
                tap_budget_used=len(clusters),
            ),
            snapshots=[CDLSnapshot(
                index=0,
                ue_velocity_mps=profile.ue_velocity_mps,
                clusters=clusters,
            )],
        )
