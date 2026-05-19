# Design: P2-7 — Non-Ring Chamber Probe Geometry Support

**Status**: DRAFT
**Parent plan**: [strict-pfs-rollout.md](./strict-pfs-rollout.md)
**Phase**: 8 (matches MIMO-First label "P2-7" for cross-repo coherence)
**Created**: 2026-05-19
**Branch**: `feature/p2-7-non-ring-chamber`
**Trigger**: MIMO-First `docs/architecture/channel-engine-data-flow.md` (2026-05-19) explicit cross-repo work item

---

## 1. Context

MIMO-First wrote a cross-repo architecture note (`channel-engine-data-flow.md` §"ring-only silent constraint") flagging a silent failure mode we own:

```python
# mimo_ota_simulator/simulator.py:13-24 (current)
def _calculate_probe_positions(self, chamber_config):
    if chamber_config.distribution == 'ring':
        angles = np.arange(num_probes) * (2 * np.pi / num_probes)   # ★ ring assumption
        x_coords = radius * np.cos(angles)
        # ...
    else:
        raise NotImplementedError(...)
```

```python
# mimo_ota_simulator/simulator.py:340 _calculate_weights_for_cluster
probe_angles_rad = np.linspace(0, 2*np.pi, num_physical_probes, endpoint=False)  # ★ ring again
```

**Silent failure**: Non-ring chamber (sparse, sector, dual-ring) gets passed through with `distribution='ring'` (or no validation that probe_positions matches reality); ChannelEgine silently uses `(port-1) × 360°/N` azimuth assumption; `.asc` reflects fictitious angles. **No error, no warning.**

MIMO-First DB already stores `Probe.position: {azimuth, elevation}` per probe (PAS rotation reads it). They just haven't passed it through the HTTP payload. Their note says:

> 主要工作在 ChannelEgine (PAS / cluster→port 映射代码读真实 probe_positions 角度, 不再按 (port-1)×360°/N 算), MIMO-First 这边是 schema + DB plumbing 的小工作

We're the bottleneck.

## 2. Scope

**In-scope**:
- `ProbePosition` Pydantic model (per-probe azimuth + elevation)
- `ChamberConfig.probe_positions: Optional[List[ProbePosition]]`
- `ChamberConfig.distribution`: extend `Literal['ring', 'multi-ring']` → `Literal['ring', 'multi-ring', 'custom']`
- `_calculate_probe_positions`: handle `distribution='custom'` (and `'multi-ring'` as alias of custom with annular structure if needed)
- `_calculate_weights_for_cluster`: read real azimuths from `probe_positions` when set
- Backward compat: `distribution='ring'` default + `probe_positions=None` → behavior unchanged (bit-identical to pre-P2-7)
- Tests: schema validation + ring regression + custom geometry
- Docs: API + SPEC v2.0 + CLAUDE.md cross-project status

**Out-of-scope**:
- Per-probe radius (MIMO-First DB stores only az/el, radius is chamber-wide via `chamber.radius_m`)
- Elevation-aware probe weights (current weights are azimuth-only; ZoA-aware weighting is P3 follow-up)
- `multi-ring` independent enum implementation (treated as alias of `custom` for now; can split later if needed)
- MIMO-First DB plumbing (their side, P2-7 their half)

## 3. Schema

```python
class ProbePosition(BaseModel):
    azimuth_deg: float = Field(..., ge=-360.0, le=360.0,
        description="Azimuth angle in horizontal plane, measured from +x axis CCW")
    elevation_deg: float = Field(0.0, ge=-90.0, le=90.0,
        description="Elevation angle from horizontal plane. 0=horizontal (default).")

class ChamberConfig(BaseModel):
    num_probes: int
    distribution: Literal['ring', 'multi-ring', 'custom'] = ...  # extend
    radius_m: float                                # uniform across all probes
    probe_positions: Optional[List[ProbePosition]] = None  # NEW
    # ... existing fields unchanged

    @model_validator(mode='after')
    def _validate_probe_positions(self):
        if self.distribution in ('custom', 'multi-ring'):
            if self.probe_positions is None:
                raise ValueError(...)
            if len(self.probe_positions) != self.num_probes:
                raise ValueError(...)
        elif self.distribution == 'ring' and self.probe_positions is not None:
            # Tightened post-PR #9 Codex P2: reject ambiguous combo.
            # Original loose policy (allow + ignore) caused silent B-matrix
            # vs cluster-weight inconsistency. Single source of truth: pick one.
            raise ValueError("ring + probe_positions is ambiguous; pick one.")
        return self
```

## 4. Math (sphere → Cartesian)

```
x = radius_m * cos(el) * cos(az)
y = radius_m * cos(el) * sin(az)
z = radius_m * sin(el)
```

For `el=0` (default) this reduces exactly to the ring formula `(r cos az, r sin az, 0)`. Backward compat preserved by construction.

## 5. `_calculate_weights_for_cluster` adaptation

```python
if chamber_config.probe_positions is not None:
    probe_angles_rad = np.deg2rad([p.azimuth_deg for p in chamber_config.probe_positions])
else:
    probe_angles_rad = np.linspace(0, 2*np.pi, num_physical_probes, endpoint=False)  # ring default
```

For `min_as_rad` (Gaussian width floor to ensure non-zero weight on at least one probe), keep `2π/N` as the "effective spacing" approximation. Works for non-uniform layouts too (mean spacing); refine later if real uneven layouts show artifacts.

## 6. Tests

| Test | What it pins |
|---|---|
| `test_ring_default_unchanged` | `distribution='ring'` + no `probe_positions` → bit-identical Cartesian positions vs pre-P2-7 |
| `test_custom_distribution_requires_positions` | `distribution='custom'` + `probe_positions=None` → ValidationError |
| `test_custom_positions_count_mismatch` | `len(probe_positions) != num_probes` → ValidationError |
| `test_custom_horizontal_plane` | Custom azimuths only, el=0 → Cartesian positions correct |
| `test_custom_with_elevation` | el != 0 → z component non-zero, magnitude = `radius * sin(el)` |
| `test_weights_use_real_azimuths` | Cluster AoA matches a specific probe's actual azimuth (not ring-linspace position) → that probe gets highest weight |
| `test_ring_regression_strict_pfs` | Existing TestRayRegression golden still bit-identical after refactor |

## 7. Sub-task list

| ID | Task | Est. |
|---|---|---|
| T8.1 | `ProbePosition` model + `ChamberConfig` schema extension + validator | 1h |
| T8.2 | `_calculate_probe_positions` custom branch + tests | 1h |
| T8.3 | `_calculate_weights_for_cluster` azimuth source switch + tests | 1h |
| T8.4 | Full regression sweep + new custom geometry e2e tests | 1h |
| T8.5 | Docs: CLAUDE.md cross-project status, API doc, SPEC v2.0 schema, rollout plan Phase 8 entry | 1h |

**Total**: ~5h (single batch).

## 8. Open questions

None — scope is concrete from MIMO-First's architecture note. Defaults all pinned.

## Changelog

- 2026-05-19: Initial draft
