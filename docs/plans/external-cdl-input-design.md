# Design: Phase 5 — External CDL Input API

**Status**: DRAFT — pending review
**Parent plan**: [strict-pfs-rollout.md](./strict-pfs-rollout.md) (Phase 4 defer 到此 Phase 之后)
**Phase**: 5
**Created**: 2026-05-18
**Branch**: `feature/external-cdl-input`
**Depends on**: Phases 0-3 merged (PR #1, #2, #3, #4)

---

## 1. Context

### Driver

MIMO-First 的 Ray-Tracing 模块输出完整 CDL 参数（angles, power, delay, XPR, K-factor, per-ray initial phases），需要直接喂给 Channel Engine 产 OTA 权重 + ASC 文件。

**当前 (post-PR #4) 状态**：library 路径 (`CustomJSONBuilder`) 已能接受外部 CDL JSON + `strict_pfs` + Propsim DL ASC 端到端跑通（实测 e2e: 3 clusters → 16 .asc files, 22246 bytes）。**但有 4 个 gap**：

| Gap | 现状 | 影响 |
|---|---|---|
| Gap 1 | API 仅接 JSON 文件路径，不接 dict / Python obj | HTTP service caller 必须写临时文件 |
| Gap 2 | Schema 简化：无 XPR、K-factor、per-ray init phases | 不能表达完整 CDL |
| Gap 3 | 没 e2e tests + 没正式 schema 文档 | 易回归、易写错 |
| Gap 4 | `doc/MIMO_First_Integration_Spec.md` v1.0 跟现状不一致且不完整 | 跨 repo 协作混乱 |

### Scope（per Q1.A）

**In-scope (本 repo)**:
- Pydantic schema (`CustomCDLProfile`, `CDLClusterSpec`)
- Factory API: `CustomCDLProfile.from_dict(d)` + `CustomCDLProfile.from_file(path)`
- Synthesizer extensions to USE XPR / K-factor / per-ray init phases (per Q2 "全部真用")
- SPEC v2.0 重写 (library-level schema doc)
- e2e tests + API doc 更新

**Out-of-scope (MIMO-First 自己做, per D11)**:
- HTTP service / FastAPI endpoint
- `synthesize_hardware_pipeline` REST handler
- Zip/Base64 packaging
- Microservice deployment

### Non-goal

- 不改 strict_pfs / ray mode 数学契约（per-realization 独立 fading 仍成立）
- 不实现 `run_with_external_clusters` 别名方法 (D11)
- 不动 F64 .asc 文件格式

---

## 2. ⚠️ Phase 5 scope 重大决策点

Q2 答 "**全部真用** XPR + K-factor + per-ray init"。XPR 真用 = **dual-polarization synthesizer 重构**，工作量 5-8 工作日，跟整个 Phase 1+2 加起来差不多。**Reviewer 先确认 commitment**。

### Sub-task XPR 真用工作量 breakdown

| 子模块 | 当前状态 | 改造工作量 |
|---|---|---|
| `_synthesize_b_matrix` dual_pol stub (`b_matrix[r, 2p]=b_matrix[r, 2p+1]` 同响应) | "假" dual-pol，无 H/V 分离 | 实现真 H/V probe response + cross-pol leakage 模型 (~2 天) |
| `g_tx` Tx steering | scalar phase per Tx antenna，无极化维 | 扩展为 (2, N_tx) H/V Tx ports (~1 天) |
| Cluster-level XPR mixing | 不存在 | 加 per-cluster 2×2 极化耦合矩阵 (~2 天) |
| strict_pfs synthesis | single-pol scalar | 适配 dual-pol broadcasting (~1 天) |
| Tests | 无 dual-pol regression | 加 H/V 极化 contract tests (~1 天) |

### 选项 A: 一次性全做 (Phase 5 完整)

Total ~5-8 天。Risk: 大 refactor，strict_pfs 数学行为可能微变（需要新 regression baseline）。

### 选项 B: 拆 Phase 5 + Phase 6

- **Phase 5** (本文档): Pydantic schema + factory API + K-factor LOS boost + per-ray init phases plumbing。XPR 字段接受存储但 synthesizer 不真用，加 `DeprecationWarning` 或 `NotImplementedError` 提示 "XPR plumbed in Phase 5, semantically applied in Phase 6"。工作量 2-3 天。
- **Phase 6** (单独): XPR / dual-pol synthesizer 重构 + tests + API doc 更新。工作量 5-8 天。

**推荐 选项 B**: 保 Phase 5 manageable，让 dual-pol 有专门的设计阶段（dual-pol 是数学契约级别改动，值得独立 review + sign-off）。

下面继续按 **选项 B (Phase 5 = schema + K-factor + per-ray init plumbing, XPR 仅 plumb)** 设计。如果你要选项 A，本文档要补 §6.x 的 dual-pol 全部细节。

---

## 3. Schema 设计 (v2.0)

### CustomCDLProfile (top-level Pydantic model)

```python
class CustomCDLProfile(BaseModel):
    # Global parameters
    center_frequency_hz: float = Field(..., gt=0)
    pathloss_db: float = Field(..., description="Free-space + shadowing path loss in dB")
    ue_velocity_mps: Tuple[float, float, float] = Field(
        ..., description="UE velocity vector in m/s (x, y, z); use units consistent with antenna positions"
    )
    is_los: bool = Field(...)

    # Per-cluster CDL definition
    clusters: List[CDLClusterSpec] = Field(..., min_length=1)

    # Optional global K-factor (overrides cluster-level k_n if present)
    k_factor_db: Optional[float] = Field(
        None,
        description="Global Ricean K-factor in dB. If set, applies to first cluster (LOS path) per TR 38.901 §7.5."
    )

    @classmethod
    def from_file(cls, path: str) -> "CustomCDLProfile": ...
    @classmethod
    def from_dict(cls, d: dict) -> "CustomCDLProfile": ...

class CDLClusterSpec(BaseModel):
    # Geometry
    delay_s: float = Field(..., ge=0)
    power_linear: float = Field(..., gt=0, description="Linear power (sum of cluster powers normalizes to 1)")
    aoa_deg: float
    aod_deg: float
    zoa_deg: float = Field(90.0, description="Default elevation = 90 (horizontal plane)")
    zod_deg: float = Field(90.0)
    as_aoa_deg: float = Field(0.0, ge=0)
    as_aod_deg: float = Field(0.0, ge=0)
    as_zoa_deg: float = Field(0.0, ge=0)
    as_zod_deg: float = Field(0.0, ge=0)

    # NEW in v2.0: polarization + ray-level
    xpr_db: Optional[float] = Field(
        None,
        description="Cross-Polarization Ratio in dB. Phase 5: PLUMBED (stored in EngineCluster.k_n_xpr); Phase 6: SEMANTICALLY APPLIED in synthesizer."
    )
    initial_phases_rad: Optional[List[float]] = Field(
        None, min_length=4, max_length=4,
        description="Per-ray polarization init phases [Φ_θθ, Φ_θφ, Φ_φθ, Φ_φφ] in radians. CDL spec format."
    )
    num_rays: int = Field(20, description="Number of sub-rays per cluster (default 20 per TR 38.901)")
```

### v1.0 vs v2.0 schema 差异

| 字段 | v1.0 (deprecated) | v2.0 |
|---|---|---|
| ue_velocity_kph | `[15, 0, 0]` or scalar `15` | `ue_velocity_mps: Tuple[float, float, float]` (单位 m/s, 显式) |
| Top-level structure | 嵌套 (`ray_tracing_channel.clusters`) 或扁平 (`global_params + clusters`) | 扁平 (`center_frequency_hz, pathloss_db, ue_velocity_mps, is_los, clusters[]`) |
| K-factor | 无 | `k_factor_db: Optional[float]` 顶层 |
| XPR | 无 | `clusters[].xpr_db: Optional[float]` |
| Per-ray init phases | 无 | `clusters[].initial_phases_rad: Optional[List[float]]` (4 个 polarization phases) |
| zoa/zod | 默认值 | 同 (Field default 90°) |
| `num_rays` | 无 | 显式可配 (默认 20) |

---

## 4. Synthesizer 改动 (per Q2 真用)

### K-factor (Phase 5 实现)

3GPP TR 38.901 §7.5.6: 对 LOS 场景，第一个 cluster (LOS path) 功率为 `K/(K+1)`，其余 cluster 总功率为 `1/(K+1)`。

代码改动位置：
- 在 `Standard3GPPBuilder` / `CustomJSONBuilder` build 后、return profile 前，加一个 `_apply_k_factor_los_boost(clusters, k_db)` 步骤
- 若 `k_factor_db is None`，无操作（保 backward compat）
- 若 `k_factor_db != None and is_los`，按 TR 38.901 公式重新分配 cluster.power

### Per-ray init phases (Phase 5 plumb, semantic 在 single-pol 下简化)

CDL 标准 4-pol phase tuple `[Φθθ, Φθφ, Φφθ, Φφφ]` 是 polarization 矩阵的相位。Single-pol 模式下 (Phase 5 范围) 只有 V-V (`Φθθ`) 有意义；H/V cross-pol 项 (`Φθφ, Φφθ, Φφφ`) 在 dual-pol 实现 (Phase 6) 才被使用。

代码改动：
- `EngineCluster.initial_phases` 接受 v2.0 schema 提供的 4-元素 list
- strict_pfs synthesizer 优先使用 `cluster.initial_phases[0]` (Θθθ) 作为 ray-level base phase 代替 `np.random.uniform`
- 注意: strict_pfs 的 **per-probe random phase** 跟 **CDL init phase** 是叠加关系，不是替换
  - strict_pfs: `phi_rnd[p] ~ U[-π, π]` 给 per-realization 独立性
  - CDL init phase: `Φθθ` 给 per-ray polarization-deterministic base
  - 最终: `A_ray[p, t] = W_base[p, t] · exp(j·(Φθθ + phi_rnd[p]))`

### XPR (Phase 5 plumb-only, Phase 6 真用)

- `CDLClusterSpec.xpr_db` 接受 + 存到 `EngineCluster.xpr_linear` (新加字段)
- synthesizer 不读这字段；如果用户期待 dual-pol 行为，Phase 5 文档明确说 "XPR is plumbed but not yet applied; see Phase 6 roadmap"

---

## 5. Python API 设计

### CustomCDLProfile 单一 source of truth

```python
# 路径 A: 从文件
profile = CustomCDLProfile.from_file('/path/to/cdl.json')

# 路径 B: 从 dict (HTTP service caller 友好)
profile = CustomCDLProfile.from_dict({
    'center_frequency_hz': 2.6e9,
    'pathloss_db': 85.0,
    'ue_velocity_mps': [4.17, 0.0, 0.0],
    'is_los': True,
    'k_factor_db': 10.0,
    'clusters': [
        {'delay_s': 0.0, 'power_linear': 0.7, 'aoa_deg': 10, 'aod_deg': -15,
         'as_aoa_deg': 5.0, 'as_aod_deg': 2.0,
         'xpr_db': 7.0, 'initial_phases_rad': [0.3, -1.2, 0.5, 2.1]},
        ...
    ],
})

# 同一 profile 喂给现有 builder
config = TargetChannelConfig(
    input_mode='custom',
    custom_profile=profile,  # NEW: pass Pydantic object directly
    # custom_profile_path: 保留作 backward compat
    ...
)
```

### TargetChannelConfig 改动

```python
class TargetChannelConfig(BaseModel):
    input_mode: Literal['standard', 'custom'] = 'standard'
    # OLD: file path only
    custom_profile_path: Optional[str] = None
    # NEW: Pydantic object (preferred)
    custom_profile: Optional[CustomCDLProfile] = None
    ...
```

### CustomJSONBuilder → CustomCDLBuilder

重命名 + 重构：
```python
class CustomCDLBuilder(ChannelProfileBuilder):
    def __init__(self, profile: CustomCDLProfile): ...  # NEW: takes object not path
    def build(self) -> UniversalChannelProfile: ...

# Backward compat shim
class CustomJSONBuilder(CustomCDLBuilder):
    """Deprecated: use CustomCDLBuilder(CustomCDLProfile.from_file(path))."""
    def __init__(self, json_filepath: str):
        warnings.warn("CustomJSONBuilder is deprecated; use CustomCDLBuilder with CustomCDLProfile.from_file()", DeprecationWarning)
        super().__init__(CustomCDLProfile.from_file(json_filepath))
```

---

## 6. SPEC v2.0 doc rewrite

替换 `doc/MIMO_First_Integration_Spec.md` 内容：

1. 标记 v1.0 deprecated（保留 v1.0 一节作历史记录）
2. v2.0 标 status: "MIMO-First service-side adapter contract" —— 服务侧 wrapper 怎么把它的 HTTP payload 转成 `CustomCDLProfile.from_dict(...)`
3. **本 repo 库层面**契约用 `CustomCDLProfile` Pydantic schema 作 single source of truth
4. 跨 repo coordination: 微服务 (channel-engine-service 在 MIMO-First repo) 自己负责把 HTTP → CustomCDLProfile.from_dict() → MIMO_OTA_Simulator.run() → ASC zip → HTTP response

---

## 7. 测试策略

| 测试 | 类型 | 覆盖 |
|---|---|---|
| `test_custom_cdl_profile_schema.py` | unit | Pydantic validation, from_file/from_dict, missing required fields, type coercion |
| `test_k_factor_los_boost.py` | unit | k_factor_db=10 时第一 cluster 功率正确 |
| `test_per_ray_init_phases.py` | unit | initial_phases_rad 真传到 strict_pfs synthesizer base phase |
| `test_xpr_plumbing.py` | unit | xpr_db 存到 EngineCluster.xpr_linear；Phase 5 不应用 |
| `test_custom_cdl_e2e.py` | integration | from_dict → MIMO_OTA_Simulator.run('strict_pfs') → Propsim DL ASC → 字节数合理 |
| `test_v1_to_v2_migration.py` | compat | v1.0 JSON 文件读入 v1.0 path 仍工作 (DeprecationWarning) |

---

## 8. Sub-task 分解 + 排序

| ID | 任务 | 依赖 | 工作量 |
|---|---|---|---|
| **T5.1** | `CustomCDLProfile` + `CDLClusterSpec` Pydantic schema 定义 + unit tests | 无 | 0.5d |
| **T5.2** | `from_dict` / `from_file` factory + edge case tests | T5.1 | 0.3d |
| **T5.3** | `CustomCDLBuilder` (new) + `CustomJSONBuilder` deprecation shim | T5.2 | 0.3d |
| **T5.4** | K-factor LOS boost application + unit test | T5.1 | 0.5d |
| **T5.5** | Per-ray init phases integration to strict_pfs synthesizer + test | T5.1, T5.3 | 0.5d |
| **T5.6** | XPR field plumbing only (no synthesizer use) + clear Phase 6 marker | T5.1 | 0.2d |
| **T5.7** | `TargetChannelConfig.custom_profile` Pydantic field + builder routing | T5.3 | 0.3d |
| **T5.8** | E2E test: dict input → strict_pfs → Propsim DL ASC | T5.3, T5.4, T5.5, T5.7 | 0.5d |
| **T5.9** | `doc/MIMO_First_Integration_Spec.md` v2.0 rewrite | T5.1 done | 0.5d |
| **T5.10** | `docs/api/mimo_ota_simulator.md` 加 CustomCDLProfile section | T5.3 done | 0.3d |

**总工作量**: ~3-4 工作日 (选项 B; XPR 仅 plumb)
**Critical path**: T5.1 → T5.3 → T5.7 → T5.8

---

## 9. Risks & Mitigation

| Risk | Mitigation |
|---|---|
| K-factor 公式实现错（TR 38.901 §7.5.6 解读） | T5.4 test 引用 spec 例子手算对照 |
| Per-ray init phases 跟 strict_pfs RNG 叠加破坏 per-realization 0 cross-corr | T5.5 加 unit test 复用 T2.1 contract，验证叠加后仍 mean off-diag → 0 |
| Pydantic v1 vs v2 兼容 (existing models 用 class-based config) | 用 ConfigDict 写新 model，旧 model 不动 |
| Backward compat: 老 JSON 文件 (温柔 v1.0 schema) | T5.3 shim + DeprecationWarning |
| 微服务 (MIMO-First side) 跟新 schema 对不上 | T5.9 SPEC v2.0 写清；本 repo 不管微服务但提供 contract |

---

## 10. Open Questions

- [ ] OQ1: 选项 A (一次性 XPR 真用) vs 选项 B (Phase 5 only plumb + Phase 6 dual-pol) —— **推荐 B**，等你确认
- [ ] OQ2: `EngineCluster` 加新字段 `xpr_linear` / `pol_initial_phases` —— OK 还是改 separate sub-model？
- [ ] OQ3: SPEC v2.0 是否同时给 MIMO-First reviewer 看？还是先在本 repo 落地再同步？
- [ ] OQ4: 选项 B 下 XPR 字段 plumbed 但不用，应 emit warning 提示用户吗？

---

## 11. Out of Scope（明确 Phase 5 不做）

- HTTP service / FastAPI endpoint
- Zip/Base64 packaging
- Dual-pol synthesizer (Phase 6 if 选 B)
- Microservice deployment
- Phase 4 cross-repo sync (defer until Phase 5 done)
- 改 F64 .asc 文件格式

---

## Changelog

- 2026-05-18: Initial design draft (Claude + @swang430 待 review)
