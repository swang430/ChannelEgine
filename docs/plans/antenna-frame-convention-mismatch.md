# Issue: Antenna LCS base-frame convention mismatch (channel-model y-axis vs OTA x-axis)

**Status**: OPEN (recorded, not yet fixed)
**Tracking issue**: [swang430/ChannelEgine#13](https://github.com/swang430/ChannelEgine/issues/13)
**Created**: 2026-06-10
**Standard**: TR 38.901 V19.0.0 (current repo parameter version)
**Found during**: branch `fix/audit-batch-1-silent-ignored-fields`, while plumbing DUT array
orientation (Sec 7.1.3 bearing/downtilt/slant) into the OTA chamber B-matrix.
**Severity**: latent correctness — silent ~90° azimuth offset between the *target channel*
geometry and the *chamber steering* geometry. Masked today because static OTA conformance
does a 360° turntable average; becomes real for **dynamic, non-rotating MIMO OTA**.

---

## 1. Symptom

The two antenna-position helpers place a ULA along **different base axes** in their local
coordinate system (LCS), then both apply the *same* `apply_orientation(bearing, downtilt,
slant)` rotation on top:

| Helper | Base ULA layout | Code |
|---|---|---|
| `channel_model_38901/common/geometry.py::get_antenna_positions` | along **+y axis**, from origin | `pos = [0, c·d_h, r·d_v]` (geometry.py:34) |
| `mimo_ota_simulator/simulator.py::_calculate_dut_antenna_positions` | along **x axis**, centered | `x = linspace(-half, +half)` (simulator.py:166) |

Both call `apply_orientation(...)` (geometry.py) with the identical Euler triple
`R = Rz(α)·Ry(β)·Rx(γ)`, α=bearing, β=downtilt, γ=slant (Sec 7.1.3).

## 2. Why LCS→GCS rotation does NOT fix it

`apply_orientation` is a *rotation operator* applied to a base frame; it is not a frame
*unifier*. Feeding the same angles into two different base frames yields two different world
layouts. Decomposing the difference between the two bases:

1. **y-axis vs x-axis** → an implicit **90° rotation about z** (azimuth). With `bearing_deg=0`,
   the channel-model boresight points +y while the OTA boresight points +x. This term enters
   relative element phase and is a **real physical discrepancy**.
2. **from-origin vs centered** → a pure global translation. Relative phase between elements is
   translation-invariant, so this term is **harmless**.

Net: only the 90° azimuth term matters, and it is *not* removed by the orientation rotation —
the rotation is defined relative to each helper's own (different) zero.

## 3. Where it bites

In `mimo_ota_simulator/simulator.py`:
- `synthesize_ota` (line ~561-562) and `_run_wifi` (line ~487-488) both build steering geometry
  via `_calculate_dut_antenna_positions` → **OTA x-frame**.
- The *target* channel (what we are trying to reproduce) comes from `Standard3GPPBuilder`
  → `ChannelSimulator` → `get_antenna_positions` → **channel-model y-frame**.

So the channel we synthesize and the channel we target are expressed in frames that differ by
~90° in azimuth. For a 360°-rotation conformance test this averages out (orientation is
integrated over); for a fixed-orientation dynamic run it does not.

## 4. Fix options (not yet chosen)

- **A. Unify the base frame.** Make `_calculate_dut_antenna_positions` use the same +y base
  axis as `get_antenna_positions` (keep the centered vs from-origin choice — it's harmless, or
  also unify for tidiness). Smallest semantic change; risk = it shifts existing OTA numerics, so
  needs a golden-regression refresh and a note that pre-existing non-zero-orientation OTA runs
  change.
- **B. Single shared helper.** Collapse both into one `get_antenna_positions` call (pass
  `centered=` flag if needed) so there is exactly one convention. Cleanest long-term; larger diff.
- **C. Document + adapter.** Keep both frames, insert an explicit 90° azimuth adapter where the
  OTA layer consumes the channel-model target, and assert the convention in tests. Lowest blast
  radius, but leaves two conventions alive.

Recommendation leaning **B** (one helper, one convention) once a golden-regression refresh is
scheduled; **A** if we want the minimal touch first.

## 5. Guardrail to add with the fix

A unit test asserting that, for `bearing=downtilt=slant=0`, a ULA's element azimuths are
identical between `get_antenna_positions` and `_calculate_dut_antenna_positions` (up to the
harmless global translation). This pins the convention so it can't silently re-diverge.
