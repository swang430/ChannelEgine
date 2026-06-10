import numpy as np
from typing import Literal, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator
from channel_model_38901.common.data_models import EngineCluster, LargeScaleParameters
from mimo_ota_simulator.cdl_schema import CustomCDLProfile

# --- V2.0 Data Models ---

class AntennaArrayConfig(BaseModel):
    """Configuration for a single antenna array (Tx or Rx)."""
    array_type: Literal['ULA', 'URA'] = Field('ULA', description="Uniform Linear or Rectangular Array")
    element_pattern: Literal['Isotropic', 'TR38.901'] = Field('Isotropic', description="Antenna element pattern")
    num_rows: int = Field(1, gt=0, description="Number of rows in the antenna array")
    num_cols: int = Field(1, gt=0, description="Number of columns in the antenna array")
    spacing_h: float = Field(0.5, description="Horizontal spacing in multiples of wavelength")
    spacing_v: float = Field(0.5, description="Vertical spacing in multiples of wavelength (for URA)")
    polarization: Literal['V', 'H', '+45', '-45'] = Field(
        'V',
        description=(
            "Antenna polarization, applied uniformly to every element of this array. "
            "'V' = vertical (θ-polarized, slant 0°), 'H' = horizontal (φ-polarized, "
            "slant 90°), '+45'/'-45' = ±45° slant cross-polarized elements. Modeled as "
            "a [θ, φ] basis vector [cos ζ, sin ζ]. Used by the dual-polarization "
            "synthesizer when ChamberConfig.dual_polarized=True; slant pols require "
            "dual_polarized=True. Default 'V' preserves single-pol behavior."
        ),
    )
    # Array orientation in the GCS (TR 38.901 Sec 7.1.3). All 0 => identity, so the
    # standard 360° turntable test (where DUT orientation is averaged out) is
    # unaffected. The fixed starting orientation only becomes meaningful for
    # dynamic (non-rotating) MIMO OTA, where it shapes the DUT-array geometry in
    # the chamber (B-matrix) and the target-channel per-antenna phases.
    bearing_deg: float = Field(0.0, description="Array bearing angle alpha (azimuth) in degrees.")
    downtilt_deg: float = Field(0.0, description="Array downtilt angle beta (elevation) in degrees.")
    slant_deg: float = Field(0.0, description="Array slant angle gamma (rotation about boresight) in degrees.")

class TargetChannelConfig(BaseModel):
    """
    Comprehensive configuration for the target wireless channel to be emulated.
    """
    # Scenario & Geometry
    input_mode: Literal['standard', 'custom'] = Field('standard', description="Input mode: standard 3GPP or custom CDL profile")
    custom_profile_path: Optional[str] = Field(None, description="DEPRECATED: legacy v1.0 JSON file path. Use custom_profile (Pydantic object) instead.")
    custom_profile: Optional[CustomCDLProfile] = Field(None, description="Preferred Phase 5+ entry: validated CustomCDLProfile object. Takes precedence over custom_profile_path when both are set.")
    standard: Literal['3gpp', 'wifi'] = Field('3gpp', description="Channel standard: 3gpp or wifi")
    model_name: str = Field('UMa', description="Name of the standard channel model (scenario or TGax model)")
    cluster_model_name: Optional[str] = Field('Stochastic', description="Name of the small-scale cluster model")
    center_frequency_hz: float = Field(2.4e9, description="Center frequency in Hz")
    bandwidth_hz: float = Field(20e6, description="Channel bandwidth in Hz (WiFi)")
    bs_position: Tuple[float, float, float] = Field((0, 0, 25), description="3D position of the Base Station")
    ue_position: Tuple[float, float, float] = Field((100, 0, 1.5), description="3D position of the User Equipment")
    ue_velocity: Tuple[float, float, float] = Field((0, 0, 0), description="3D velocity vector of the UE in m/s")

    # Antenna Config
    tx_antenna: AntennaArrayConfig = Field(default_factory=AntennaArrayConfig)
    rx_antenna: AntennaArrayConfig = Field(default_factory=AntennaArrayConfig)

    # Channel Condition
    force_condition: Optional[Literal['LOS', 'NLOS']] = Field(None, description="Force Line-of-Sight or Non-Line-of-Sight")
    random_seed: Optional[int] = Field(None, description="Seed for the random number generator")
    use_median_lsps: bool = Field(False, description="Use median LSP values instead of random sampling")

    # TR 38.901 CDL compliance options (opt-in; default off preserves raw-table CDL).
    scale_to_scenario: bool = Field(False, description="Rescale tabulated CDL delays/angles to the scenario LSPs (TR 38.901 Sec 7.7.3 / 7.7.5.1)")
    split_subclusters: bool = Field(False, description="Split the 2 strongest clusters into 3 sub-clusters (TR 38.901 Sec 7.7.1)")
    c_ds_ns: float = Field(3.91, description="Sub-cluster delay-spread constant c_DS in ns (used when split_subclusters=True)")

    # WiFi Options
    enable_ofdm: bool = Field(False, description="Enable WiFi OFDM subcarrier generation")
    n_fft: Optional[int] = Field(None, description="WiFi OFDM FFT size")
    phase_policy: Literal['zero', 'random'] = Field('zero', description="WiFi tap phase policy")
    normalize_power: bool = Field(True, description="Normalize WiFi tap power to sum to 1")
    delay_scale: Optional[float] = Field(None, description="WiFi tap delay scaling factor")

    # Reciprocity Options
    enable_ul_dl: bool = Field(False, description="Enable UL/DL simultaneous generation")
    reciprocity_transform: Literal['transpose', 'conjugate_transpose'] = Field(
        'conjugate_transpose',
        description="Reciprocity transform for UL channel"
    )

    @model_validator(mode='after')
    def _validate_wifi_input_mode(self):
        """Audit F4 (2026-05-19): reject standard='wifi' + input_mode='custom'.

        The simulator's run() branches on standard=='wifi' BEFORE
        create_builder() is consulted (simulator.py L435), so the WiFi path
        regenerates clusters from WiFiChannelSimulator and silently drops
        any custom_profile / custom_profile_path the caller supplied. Users
        who fed a CustomCDLProfile expecting CDL-style synthesis on WiFi
        got a Tgax-derived channel with no error / warning.

        WiFi custom-profile support is not implemented (the WiFi path lacks
        a CustomCDLBuilder branch). Until that is wired up, fail fast at
        config-construction time rather than silently honoring 'wifi' over
        'custom'.
        """
        if self.standard == 'wifi' and self.input_mode == 'custom':
            raise ValueError(
                "standard='wifi' + input_mode='custom' is not supported: the "
                "WiFi path in MIMO_OTA_Simulator._run_wifi() regenerates "
                "clusters from WiFiChannelSimulator and silently ignores "
                "custom_profile / custom_profile_path. Either set "
                "standard='3gpp' to use CustomCDLBuilder, or set "
                "input_mode='standard' to use the WiFi Tgax model."
            )
        return self

class CalibrationEntry(BaseModel):
    """Calibration data for a single RF chain (Cable + Attenuator + Probe)."""
    port_id: int = Field(..., description="F64 Output Port ID (1-based)")
    cable_loss_db: float = Field(0.0, description="Fixed cable loss")
    cable_phase_deg: float = Field(0.0, description="Fixed cable phase offset")
    probe_gain_dbi: float = Field(0.0, description="Probe antenna gain")
    # Future: Add attenuation phase curve LUT here if needed

class CalibrationConfig(BaseModel):
    """Collection of calibration data for the entire chamber."""
    entries: List[CalibrationEntry] = Field(..., description="List of calibration entries per port")
    
    def get_system_response(self, port_id: int) -> complex:
        """Returns the complex system response factor (Cable + Probe) for a given port."""
        for entry in self.entries:
            if entry.port_id == port_id:
                loss_linear = 10**(-entry.cable_loss_db / 10.0)
                gain_linear = 10**(entry.probe_gain_dbi / 10.0)
                total_gain_mag = np.sqrt(loss_linear * gain_linear)
                phase_rad = np.deg2rad(entry.cable_phase_deg)
                return total_gain_mag * np.exp(1j * phase_rad)
        return 1.0 + 0j

    def get_correction_factor(self, port_id: int) -> complex:
        """Returns the complex correction factor (Pre-distortion) for a given port."""
        for entry in self.entries:
            if entry.port_id == port_id:
                # Total Phase Shift = Cable + Probe
                # We want to APPLY the INVERSE (Pre-distortion) -> multiply by exp(+j * phase)
                # Wait, usually we say system adds phase phi. So we generate exp(-j*phi).
                # Here we model the physical system. 
                # H_sys = G_cable * exp(j*phi_cable) * G_probe
                # We want H_eff * H_sys = H_target
                # So H_eff = H_target / H_sys
                # Correction = 1 / H_sys
                
                h_sys = self.get_system_response(port_id)
                if abs(h_sys) < 1e-9:
                    return 0j
                return 1.0 / h_sys
        return 1.0 + 0j

class PropsimExportConfig(BaseModel):
    """Configuration for exporting to Keysight Propsim F64."""
    filename: str = Field("propsim_channel.asc", description="Output filename")
    mode: Literal['A', 'B'] = Field('A', description="Amplitude Control Mode: 'A' (External ATT) or 'B' (Internal Scaling)")
    sample_rate_hz: float = Field(100e6, description="Sampling rate for the CIR time series")
    duration_s: float = Field(0.1, description="Total duration of the channel sequence")
    target_output_power_dbm: float = Field(-10.0, description="Target output power at F64 port (Mode A reference)")

# --- Shared Data Models ---

class ProbeAntennaConfig(BaseModel):
    """Configuration for the antenna of a single physical probe."""
    gain_dBi: float = Field(0.0, description="The gain of the probe antenna in dBi.")
    pattern: Literal['Isotropic', 'TR38.901'] = Field(
        'Isotropic',
        description=(
            "探头辐射方向图。'TR38.901' 在 B 矩阵中按 Table 7.3-1 形状因子"
            "(峰值 0 dB, boresight 指向暗室中心)衰减离轴方向; 峰值增益"
            "仍由 gain_dBi 单独给出, 两者不叠加 8 dBi。'Isotropic' 因子恒 1。"
        ))


class ProbePosition(BaseModel):
    """Per-probe geometric position on the chamber sphere (P2-7).

    Used when ChamberConfig.distribution != 'ring' to specify non-uniform
    probe layouts (sparse, sector, dual-ring, etc). Radius is taken from
    ChamberConfig.radius_m (uniform across all probes — matches
    MIMO-First DB schema where Probe.position is {azimuth, elevation}).
    """
    azimuth_deg: float = Field(
        ..., ge=-360.0, le=360.0,
        description="Azimuth angle in horizontal plane (degrees, CCW from +x axis).",
    )
    elevation_deg: float = Field(
        0.0, ge=-90.0, le=90.0,
        description="Elevation from horizontal plane (degrees). 0=horizontal (default).",
    )


class ChamberConfig(BaseModel):
    """
    Configuration for the physical properties of the OTA chamber.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_probes: int = Field(..., description="Number of physical probes in the chamber.")
    dual_polarized: bool = Field(False, description="Whether each probe is dual-polarized.")
    distribution: Literal['ring', 'multi-ring', 'custom'] = Field(
        ...,
        description=(
            "Probe arrangement. 'ring' = uniform horizontal-plane ring (TR 37.977 §6.1 "
            "standard, computed from num_probes + radius_m). 'multi-ring' / 'custom' "
            "require explicit probe_positions per element."
        ),
    )
    radius_m: float = Field(..., description="Radius of the probe sphere (meters, uniform).")
    probe_positions: Optional[List[ProbePosition]] = Field(
        None,
        description=(
            "Per-probe azimuth/elevation. REQUIRED when distribution != 'ring' "
            "(P2-7 non-ring support). MUST have length == num_probes. MUST be "
            "None when distribution='ring' (the uniform-ring formula is the "
            "authoritative source there; allowing both would cause silent "
            "B-matrix vs cluster-weight inconsistency — see PR #9 Codex P2)."
        ),
    )
    probe_antenna: ProbeAntennaConfig = Field(default_factory=ProbeAntennaConfig, description="The antenna specification for each probe.")
    quiet_zone_radius_m: Optional[float] = Field(
        None,
        description=(
            "Operator metadata: nominal quiet-zone radius (m) of the physical chamber. "
            "**The library does NOT validate against this field** — it is stored for "
            "documentation / report-generation purposes and downstream consumers can "
            "read it back from the config. If you need quiet-zone-aware geometry "
            "checks (e.g. probe-radius vs QZ), gate them in the calling layer "
            "(GUI / micro-service) before constructing the ChamberConfig. "
            "Audit F1 (2026-05-19): wording sharpened from 'for validation' to "
            "prevent the misread that this triggers a runtime check."
        ),
    )
    calibration_vector: Optional[np.ndarray] = Field(None, description="Complex vector for amplitude and phase calibration.")

    @model_validator(mode='after')
    def _validate_probe_positions(self):
        """Enforce P2-7 schema consistency between distribution and probe_positions.

        Strict rule (per PR #9 Codex P2): the two fields cannot disagree.
          - distribution='ring'      => probe_positions MUST be None
          - distribution='custom'/'multi-ring' => probe_positions REQUIRED, length == num_probes

        The earlier looser policy (allow ring + probe_positions, ignore positions)
        caused a silent inconsistency: _calculate_probe_positions used the ring
        formula while _calculate_weights_for_cluster picked up the supplied
        azimuths — B-matrix geometry and cluster→probe weighting disagreed.
        """
        if self.distribution in ('custom', 'multi-ring'):
            if self.probe_positions is None:
                raise ValueError(
                    f"distribution={self.distribution!r} requires probe_positions to be set "
                    f"(list of {self.num_probes} ProbePosition entries). Use distribution='ring' "
                    f"if you want the legacy uniform-ring layout."
                )
            if len(self.probe_positions) != self.num_probes:
                raise ValueError(
                    f"len(probe_positions)={len(self.probe_positions)} != num_probes="
                    f"{self.num_probes}. Provide exactly one ProbePosition per physical probe."
                )
        elif self.distribution == 'ring' and self.probe_positions is not None:
            raise ValueError(
                f"distribution='ring' with probe_positions provided is ambiguous "
                f"(ring formula vs supplied positions disagree). Either set "
                f"distribution='custom' (positions authoritative) or drop "
                f"probe_positions (ring formula authoritative)."
            )
        return self

class UniversalChannelProfile(BaseModel):
    """
    A model-agnostic representation of the channel's multipath structure.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    center_frequency_hz: float = Field(..., description="Center frequency in Hz")
    ue_velocity: Any = Field(..., description="UE velocity vector (numpy array)")
    pathloss_db: float = Field(..., description="Total calculated pathloss in dB")
    clusters: List[EngineCluster] = Field(..., description="List of fully defined physical clusters")
    lsps: Optional[LargeScaleParameters] = Field(None, description="LSPs for informational/logging purposes")
    is_los: bool = Field(False, description="Whether the channel is LOS")
