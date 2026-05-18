"""External CDL input schema (Phase 5 — see docs/plans/external-cdl-input-design.md).

Pydantic v2 models for callers (MIMO-First micro-service, scripts, GUI uploads)
to pass a complete CDL profile into the OTA pipeline. Two entry points:

    profile = CustomCDLProfile.from_file('/path/to/cdl.json')
    profile = CustomCDLProfile.from_dict({...})

Then feed via ``TargetChannelConfig(input_mode='custom', custom_profile=profile)``.

Field semantics follow 3GPP TR 38.901 §7 / TR 37.977 Annex F conventions.

This schema is the **library-level contract**. The MIMO-First micro-service is
responsible for its own HTTP/JSON wrapper that ultimately produces a dict
suitable for ``CustomCDLProfile.from_dict``.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CDLClusterSpec(BaseModel):
    """One CDL cluster (multipath group), with optional polarization-aware fields.

    Phase 5 status of each field:
      - geometry + AS + power + delay: fully consumed by ray / strict_pfs synthesizer
      - xpr_db: PLUMBED only (stored in EngineCluster.xpr_linear); SEMANTIC use
        comes in Phase 6 (dual-pol synthesizer)
      - initial_phases_rad: index 0 (Θθθ) is consumed by strict_pfs as the
        per-ray base phase added to the per-probe random component; indices
        1-3 (cross-pol) are PLUMBED only, Phase 6 will use them
    """

    model_config = ConfigDict(extra='forbid')

    # ---- Geometry / power (consumed today) -------------------------------
    delay_s: float = Field(..., ge=0.0, description="Cluster delay in seconds")
    power_linear: float = Field(
        ..., gt=0.0,
        description="Linear cluster power; the builder normalizes per-cluster powers to sum to 1.",
    )
    aoa_deg: float = Field(..., description="Azimuth of Arrival (deg)")
    aod_deg: float = Field(..., description="Azimuth of Departure (deg)")
    zoa_deg: float = Field(90.0, description="Zenith of Arrival (deg); default 90 (horizontal plane)")
    zod_deg: float = Field(90.0, description="Zenith of Departure (deg); default 90")
    as_aoa_deg: float = Field(0.0, ge=0.0, description="Angle Spread of AoA (deg)")
    as_aod_deg: float = Field(0.0, ge=0.0, description="Angle Spread of AoD (deg)")
    as_zoa_deg: float = Field(0.0, ge=0.0, description="Angle Spread of ZoA (deg)")
    as_zod_deg: float = Field(0.0, ge=0.0, description="Angle Spread of ZoD (deg)")

    # ---- Polarization (Phase 5 plumb-only for XPR; Phase 6 fully applied) ---
    xpr_db: Optional[float] = Field(
        None,
        description=(
            "Cross-Polarization Ratio (XPR) in dB. PLUMBED in Phase 5: stored "
            "but not applied in synthesizer. SEMANTIC use lands in Phase 6 "
            "(dual-pol synthesizer)."
        ),
    )

    # ---- Ray-level (initial_phases_rad[0] used in Phase 5; rest plumb only) ---
    initial_phases_rad: Optional[List[float]] = Field(
        None,
        min_length=4,
        max_length=4,
        description=(
            "CDL per-ray polarization initial phases [Φ_θθ, Φ_θφ, Φ_φθ, Φ_φφ] "
            "in radians. Phase 5: index 0 (Θθθ) added to strict_pfs ray-level "
            "base phase; indices 1-3 plumbed only."
        ),
    )

    num_rays: int = Field(
        20, gt=0,
        description="Number of sub-rays per cluster (TR 38.901 default 20).",
    )


class CustomCDLProfile(BaseModel):
    """Top-level external CDL profile: global parameters + cluster list.

    Use ``from_dict`` or ``from_file`` factories; do not construct directly
    if you want validation.
    """

    model_config = ConfigDict(extra='forbid')

    # ---- Global ---------------------------------------------------------
    center_frequency_hz: float = Field(..., gt=0.0)
    pathloss_db: float = Field(
        ..., description="Free-space + shadowing path loss in dB",
    )
    ue_velocity_mps: Tuple[float, float, float] = Field(
        ...,
        description=(
            "UE velocity vector in m/s (x, y, z). Use m/s NOT kph "
            "(legacy v1.0 spec used kph; this v2.0 schema standardizes m/s)."
        ),
    )
    is_los: bool = Field(...)

    # ---- Optional global K-factor (Phase 5: applies to first cluster) ---
    k_factor_db: Optional[float] = Field(
        None,
        description=(
            "Global Ricean K-factor in dB. Applied (per TR 38.901 §7.5.6) "
            "to first cluster only when is_los=True. If None, no boost; "
            "cluster powers normalize as-is."
        ),
    )

    # ---- Clusters -------------------------------------------------------
    clusters: List[CDLClusterSpec] = Field(..., min_length=1)

    # ---- Validators ----------------------------------------------------
    @field_validator('ue_velocity_mps', mode='before')
    @classmethod
    def _coerce_velocity(cls, v):
        """Accept list or tuple of length 3."""
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return tuple(float(x) for x in v)
        raise ValueError(
            f"ue_velocity_mps must be a 3-element list/tuple of floats; got {v!r}"
        )

    @model_validator(mode='after')
    def _validate_k_factor_consistency(self):
        """Warn-only style: K-factor with is_los=False is unusual but not invalid.
        We don't raise — TR 38.901 doesn't strictly forbid; builder will just
        skip the LOS boost.
        """
        return self

    # ---- Factories ----------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> 'CustomCDLProfile':
        """Load and validate a CustomCDLProfile from a JSON file on disk."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CDL profile not found: {path}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"CDL profile {path} is not valid JSON: {e}") from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> 'CustomCDLProfile':
        """Validate a dict against the schema and return a CustomCDLProfile."""
        return cls.model_validate(d)
