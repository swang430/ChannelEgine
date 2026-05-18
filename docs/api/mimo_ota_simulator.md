# MIMO_OTA_Simulator Public API

**Module**: `mimo_ota_simulator.simulator.MIMO_OTA_Simulator`
**Stability**: `run()` signature is **stable**; return dict fields listed in §4 are stable.
**Last updated**: 2026-05-18 (Phase 1 of strict-pfs-rollout)

---

## 1. Entry point

```python
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import (
    ChamberConfig, TargetChannelConfig, AntennaArrayConfig,
)

sim = MIMO_OTA_Simulator()
result = sim.run(
    chamber_config: ChamberConfig,
    channel_config: TargetChannelConfig,
    synthesis_method: Literal['pinv', 'cluster', 'ray', 'strict_pfs'] = 'pinv',
    probe_phase_jitter: bool = False,
) -> Dict
```

`MIMO_OTA_Simulator()` takes no arguments — all configuration is passed at `run()` call time.

> **Note for `MIMO-First` / microservice consumers**: this repo deliberately does
> **not** expose `OTASimulator` (the class is named `MIMO_OTA_Simulator`), nor
> `run_with_external_clusters(...)` (the public entry is `run()`), nor a
> constructor-based config pattern. Service-side adapters should target the
> signature above (decision D11 in [strict-pfs-rollout.md](../plans/strict-pfs-rollout.md)).

---

## 2. `synthesis_method` semantics

| value | status | when to use | math contract |
|---|---|---|---|
| `'strict_pfs'` | **recommended** (new in Phase 1) | 想跳过输出端相位校准；对单 realization 精度敏感的 first-call / low-Doppler 实验 | per-(probe, cluster) 独立 fading；`E[A_i·A_j*] = 0 for i≠j` per-realization |
| `'ray'` | active default | 大平均统计验证；多 drop 收敛性研究；jitter 需要的传统场景 | Ray-level GSCM；scalar phi per ray（共享跨 probe）；strict PFS 仅 ensemble 极限成立 |
| `'cluster'` | ⚠️ **deprecated** | — | Rank-1 outer product (`np.outer(a_rx, a_tx)`)，结构性不能 strict PFS |
| `'pinv'` | ⚠️ **deprecated, broken** | — | Deterministic 伪逆映射；不产 wideband impulse response，**F64 export 直接 IndexError** |

Deprecated methods仍工作（除 pinv F64 export 外），但会触发 `DeprecationWarning`，
将在未来 release 移除。

### 推荐迁移路径

```python
# OLD
result = sim.run(chamber, channel, synthesis_method='cluster')
# NEW (drop-in replacement, more physically accurate)
result = sim.run(chamber, channel, synthesis_method='ray')
# RECOMMENDED for new code (avoids output phase calibration)
result = sim.run(chamber, channel, synthesis_method='strict_pfs')
```

---

## 3. `probe_phase_jitter` interactions

`probe_phase_jitter=True` 给每个 probe 加 ±π 均匀随机相位。**Interactions**：

| 组合 | 结果 | 表现 |
|---|---|---|
| `jitter=True` + `synthesis_method='strict_pfs'` | **redundant** | runtime `UserWarning`；UI 自动 disable |
| `jitter=True` + `chamber.calibration_vector` 提供 | **effect-wise mutex** | runtime `UserWarning`；jitter 会 swamp cal phase |
| `jitter=True` + `synthesis_method in ('ray', 'cluster')` | 有意义 | 打破几何耦合相干，需要大平均才统计有效 |
| `jitter=False` 默认 | 无相位噪声 | 推荐 (strict_pfs / 应用 cal 时) |

`probe_phase_jitter` 量级 hardcoded 为 ±π（180°）。
（**注意**：UI 文案曾误写 "±10度"，已在 Phase 0 commit `9eb1a06` 修正为 ±180°。
更小 jitter 量级不会破坏 cluster 模型相干性，所以保留 ±π 是 by-design。）

---

## 4. `run()` 返回 dict —— stable fields

```python
{
    # Core synthesis outputs ----------------------------------------------
    'transmit_mapping_matrix_A': ndarray,    # (N_probes_effective, N_tx) complex
    'effective_H_matrix':         ndarray,    # B @ A, (N_rx, N_tx) complex
    'B_matrix':                   ndarray,    # (N_rx, N_probes_effective) complex
    'target_H_matrix':            ndarray,    # (N_rx, N_tx) complex (for pinv reference)

    # Channel state -------------------------------------------------------
    'center_frequency_hz':        float,
    'pathloss_db':                float,
    'is_los':                     bool | None,
    'lsps':                       LargeScaleParameters | None,
    'clusters':                   List[EngineCluster],

    # Wideband / time-series (consumed by F64 exporter) ------------------
    'channel_impulse_response':   List[{'delay': float,
                                        'coeff': ndarray (P, Tx),
                                        'cluster_index': int}],
    'ray_components':             List[{'W_base':       ndarray (P, Tx),
                                        'phi_init':     float | ndarray (P,),
                                        'doppler_hz':   float,
                                        'cluster_index': int,
                                        'phi_axis':     'row' | 'col' (optional)}] | None,
    'probe_phase_jitter_vector':  ndarray (P,) | None,
    'probe_tx_power_contributions': ndarray,
    'cluster_probe_mapping':      Dict[int, ndarray],

    # UL mirror (only populated when channel_config.enable_ul_dl = True) -
    'ul_enabled':                  bool,
    'reciprocity_transform':       str,
    'target_H_matrix_ul':          ndarray | None,
    'transmit_mapping_matrix_A_ul': ndarray | None,
    'effective_H_matrix_ul':       ndarray | None,
    'channel_impulse_response_ul': List[...] | None,
    'ray_components_ul':           List[...] | None,
}
```

### `phi_init` data contract（important for downstream code）

- `synthesis_method='ray'` (and `cluster`, `pinv`): `phi_init` 是 Python **scalar** (`float`)
- `synthesis_method='strict_pfs'`: `phi_init` 是 numpy `(N_probes,)` **vector**
- Downstream code 应通过 `np.ndim(phi) == 0` 判别（不要用 `np.isscalar()`，
  对 0-d ndarray 误判）

UL 镜像下 `ray_components_ul[i]['phi_init'] = -ray_components[i]['phi_init']`
（element-wise negation，works for both scalar and vector）。

### `phi_axis` 字段（optional, vector phi_init only）

当 `phi_init` 是向量时（strict_pfs），exporter 需要知道它应该 broadcast 到
W_base 的哪个轴。约定：

- `'row'` — phi 对齐 `W_base` 第 0 维 (DL: probe axis)
- `'col'` — phi 对齐 `W_base` 第 1 维 (UL after transpose: probe axis)

**内部 simulator + reciprocity builder 总会设置此 tag**。

**外部 caller / 老存档 ray_components** 不带此 tag 也可以：exporter 会从
shape 推断 axis（仅在 W_base 是 square 即 `num_tx == num_probes` 时才无法
推断并抛 `ValueError` 提示需手动加 tag）。

scalar `phi_init`（ray/cluster/pinv）不需要也忽略此字段。

---

## 5. Recommended usage patterns

### Single snapshot, recommended new code

```python
sim = MIMO_OTA_Simulator()
result = sim.run(chamber, channel, synthesis_method='strict_pfs')
# Use result['effective_H_matrix'] for analysis,
#     result['channel_impulse_response'] for F64 export.
```

### Statistical validation across drops

```python
results = []
for seed in range(100):
    channel.random_seed = seed
    r = sim.run(chamber, channel, synthesis_method='strict_pfs')
    results.append(r['effective_H_matrix'])
R_mean = np.mean([H @ H.conj().T for H in results], axis=0)
```

### F64 hardware export

```python
from mimo_ota_simulator.exporters import PropsimASCIIExporter
from mimo_ota_simulator.data_models import PropsimExportConfig

cfg = PropsimExportConfig(filename='out.asc', mode='B', duration_s=0.1, sample_rate_hz=1000)
PropsimASCIIExporter(cfg).export(result, direction='dl')  # or 'ul'
```

The exporter handles both scalar (`ray`) and vector (`strict_pfs`) `phi_init`
via axis-aware broadcasting (DL `[:, None]`, UL `[None, :]`).

---

## 6. Stability guarantees

- `run()` signature: **stable** within Phase 1+ window. Adding new optional
  kwargs is allowed; removing or renaming requires deprecation cycle.
- Return dict fields listed in §4: **stable**. Adding new keys is allowed;
  removing/renaming requires deprecation cycle.
- `synthesis_method` enum: `'cluster'` and `'pinv'` are deprecated and will
  be removed in a future release. `'ray'` and `'strict_pfs'` are stable.
- `MIMO_OTA_Simulator` class name: stable. (Does NOT expose alias
  `OTASimulator`; that name is reserved.)

---

## 7. External CDL Input via CustomCDLProfile (Phase 5+)

For callers that produce their own CDL parameters (e.g. ray-tracing pipelines,
MIMO-First micro-service), bypass the built-in 3GPP scenario generator by
passing a `CustomCDLProfile` Pydantic object.

### Schema (`mimo_ota_simulator.cdl_schema`)

```python
class CustomCDLProfile(BaseModel):
    center_frequency_hz: float
    pathloss_db: float
    ue_velocity_mps: Tuple[float, float, float]   # m/s, 3-vector
    is_los: bool
    k_factor_db: Optional[float] = None            # TR 38.901 §7.5.6 LOS boost
    clusters: List[CDLClusterSpec]                  # min_length=1

class CDLClusterSpec(BaseModel):
    delay_s: float                                  # ≥0
    power_linear: float                             # >0; builder normalizes sum→1
    aoa_deg: float
    aod_deg: float
    zoa_deg: float = 90.0
    zod_deg: float = 90.0
    as_aoa_deg: float = 0.0
    as_aod_deg: float = 0.0
    as_zoa_deg: float = 0.0
    as_zod_deg: float = 0.0
    xpr_db: Optional[float] = None                  # Phase 5 plumb-only; Phase 6 dual-pol
    initial_phases_rad: Optional[List[float]] = None # [Θθθ, Θθφ, Θφθ, Θφφ], len=4
    num_rays: int = 20
```

`extra='forbid'`: typo'd / legacy v1.0 field names raise `ValidationError`.

### Two entry points

```python
# A. From a Python dict (HTTP service caller path)
profile = CustomCDLProfile.from_dict({
    'center_frequency_hz': 2.6e9, 'pathloss_db': 88.5,
    'ue_velocity_mps': [4.17, 0.0, 0.0], 'is_los': True,
    'k_factor_db': 10.0,
    'clusters': [
        {'delay_s': 0.0, 'power_linear': 1.0,
         'aoa_deg': 0.0, 'aod_deg': 0.0,
         'as_aoa_deg': 1.0, 'as_aod_deg': 0.5,
         'xpr_db': 12.0,
         'initial_phases_rad': [0.785, 0.0, 0.0, 0.0]},
        # ... more clusters
    ],
})

# B. From a JSON file on disk
profile = CustomCDLProfile.from_file('/path/to/cdl.json')
```

### Pass through `TargetChannelConfig.custom_profile`

```python
config = TargetChannelConfig(
    input_mode='custom',
    custom_profile=profile,        # preferred: Pydantic object
    # custom_profile_path: str     # legacy file-path path (deprecated)
    tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
    rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
)
result = MIMO_OTA_Simulator().run(chamber, config, synthesis_method='strict_pfs')
```

### What gets consumed vs plumbed (post-Phase 6 status)

| Field | Phase 5 | Phase 6 | Phase 7 (planned) |
|---|---|---|---|
| geometry, AS, delay, power | ✅ Consumed | ✅ | — |
| `k_factor_db` (when `is_los=True`) | ✅ Redistributes LOS cluster power per TR 38.901 §7.5.6 | ✅ | — |
| `initial_phases_rad[0]` (Θθθ) | ✅ Per-cluster base phase added to strict_pfs/ray random component | ✅ | — |
| `initial_phases_rad[1:]` (Θθφ, Θφθ, Θφφ) | 🟡 Plumbed only | ✅ Used by dual-pol pol_matrix (TR 38.901 §7.3.2) | — |
| `xpr_db` | 🟡 Plumbed only (stored as `EngineCluster.xpr_linear`) | ✅ Used by dual-pol pol_matrix off-diag sqrt(1/κ) terms | — |
| `num_rays` | ⚠️ Ignored (hardcoded 20) | ⚠️ Ignored | Configurable |

### Living example

`tests/test_custom_cdl_e2e.py` exercises the full pipeline (dict → ASC files).
`tests/test_antenna_polarization.py` covers Phase 6 dual-pol path.

---

## 8. Dual-Polarization Behavior (Phase 6+)

When `ChamberConfig.dual_polarized=True`, the synthesizer applies the
TR 38.901 §7.3.2 per-cluster 2×2 polarization mixing matrix:

```
        [exp(jΦ_θθ),             sqrt(1/κ)·exp(jΦ_θφ)]
M_c  =  [                                              ]
        [sqrt(1/κ)·exp(jΦ_φθ),   exp(jΦ_φφ)         ]

    κ = EngineCluster.xpr_linear (or default 7 dB if not set + warning)
    Φ = EngineCluster.initial_phases (or [0,0,0,0] if not set)
```

### Projection factorization

- **A-matrix side (Tx projection)** lives in the synthesizer:
  `m_col = M @ tx_pol_vec` — V probe ports (index 2p) get `m_col[0]`,
  H probe ports (index 2p+1) get `m_col[1]`
- **B-matrix side (Rx projection)** lives in `_synthesize_b_matrix`:
  V DUT antenna couples to V probe ports only; H DUT couples to H only
- **`effective_H = B · A`** then reconstructs `rx_pol_vec · M · tx_pol_vec`
  per cluster, matching the TR 38.901 channel coefficient

The A-matrix output (`'transmit_mapping_matrix_A'` /
`'probe_tx_power_contributions'`) is exposed for diagnostics and reflects
what a real chamber driver writes to each probe port — including the
H-port drive that a V DUT can't physically receive.

### Polarization × scenario behavior matrix

| `dual_polarized` | `tx.polarization` | `rx.polarization` | Channel scale | Notes |
|---|---|---|---|---|
| False | V (default) | V (default) | M[0,0] (implicit) | Phase 5 unchanged path |
| False | V | H | **raise** ValueError | OQ4: H requires `dual_polarized=True` |
| True | V | V | `exp(jΦ_θθ)` (co-pol) | Magnitude 1 per cluster |
| True | V | H | `sqrt(1/κ)·exp(jΦ_φθ)` (cross-pol) | Attenuated by sqrt(1/κ) |
| True | H | V | `sqrt(1/κ)·exp(jΦ_θφ)` | Attenuated |
| True | H | H | `exp(jΦ_φφ)` (co-pol) | Magnitude 1 |

Slant ±45° / arbitrary angle DUT polarizations are Phase 7 (deferred);
see `docs/plans/strict-pfs-rollout.md` §Phase 7.

### Strict-PFS contract under dual-pol

The Phase 1 per-realization cross-correlation contract
(`E[A_i · A_j*] = 0 for i ≠ j`) **still holds** under dual-pol synthesis.
The per-cluster pol_factor is constant across all rays in the cluster,
so the per-effective-probe random phase term still gives independent
draws across ports. Verified by `TestDualPolCrossCorrelationContract`.

---

## 9. See also

- [strict-pfs-rollout.md](../plans/strict-pfs-rollout.md) — overall multi-phase plan
- [strict-pfs-design.md](../plans/strict-pfs-design.md) — Phase 1 design rationale
- [external-cdl-input-design.md](../plans/external-cdl-input-design.md) — Phase 5 design
- `../../doc/MIMO_First_Integration_Spec.md` — cross-repo contract v2.0
- `simulator.py` source for current implementation
- `tests/test_strict_pfs.py` + `tests/test_cdl_schema.py` + `tests/test_custom_cdl_e2e.py` for behavior contracts in test form
