import numpy as np
from typing import Literal, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict
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
    pattern: Literal['Isotropic', 'TR38.901'] = Field('Isotropic', description="The radiation pattern of the probe antenna.")

class ChamberConfig(BaseModel):
    """
    Configuration for the physical properties of the OTA chamber.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_probes: int = Field(..., description="Number of physical probes in the chamber.")
    dual_polarized: bool = Field(False, description="Whether each probe is dual-polarized.")
    distribution: Literal['ring', 'multi-ring'] = Field(..., description="Arrangement of the probes.")
    radius_m: float = Field(..., description="The radius of the probe ring in meters.")
    probe_antenna: ProbeAntennaConfig = Field(default_factory=ProbeAntennaConfig, description="The antenna specification for each probe.")
    quiet_zone_radius_m: Optional[float] = Field(None, description="Radius of the chamber's quiet zone in meters for validation.")
    calibration_vector: Optional[np.ndarray] = Field(None, description="Complex vector for amplitude and phase calibration.")

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
