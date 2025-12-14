import numpy as np
from typing import Literal, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict

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
    model_name: str = Field('UMa', description="Name of the standard channel model (Large-Scale Environment)")
    cluster_model_name: Optional[str] = Field('Stochastic', description="Name of the small-scale cluster model")
    center_frequency_hz: float = Field(2.4e9, description="Center frequency in Hz")
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
                
                loss_linear = 10**(-entry.cable_loss_db / 10.0) # Power ratio < 1
                gain_linear = 10**(entry.probe_gain_dbi / 10.0) # Power ratio > 1
                total_gain_mag = np.sqrt(loss_linear * gain_linear) # Voltage ratio
                
                phase_rad = np.deg2rad(entry.cable_phase_deg)
                h_sys = total_gain_mag * np.exp(1j * phase_rad)
                
                if abs(h_sys) < 1e-9: return 0j
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

class Cluster(BaseModel):
    """
    Represents a single cluster in a channel model (simplified for OTA sim use).
    """
    aod_deg: float = Field(..., description="Azimuth of Departure in degrees.")
    aoa_deg: float = Field(..., description="Azimuth of Arrival in degrees.")
    power_dB: float = Field(..., description="Average power of the cluster in dB.")
    delay_s: float = Field(..., description="Delay of the cluster in seconds.")

class ChannelProfile(BaseModel):
    """
    A model-agnostic representation of the channel's multipath structure.
    """
    clusters: List[Cluster] = Field(..., description="List of multipath clusters.")