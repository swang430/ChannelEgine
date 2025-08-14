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
    cluster_model_name: str = Field('Stochastic', description="Name of the small-scale cluster model")
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

# --- Shared Data Models ---

class ChamberConfig(BaseModel):
    """
    Configuration for the physical properties of the OTA chamber.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_probes: int = Field(..., description="Number of physical probes in the chamber.")
    dual_polarized: bool = Field(False, description="Whether each probe is dual-polarized.")
    distribution: Literal['ring', 'multi-ring'] = Field(..., description="Arrangement of the probes.")
    radius_m: float = Field(..., description="The radius of the probe ring in meters.")
    calibration_matrix_B: np.ndarray = Field(..., description="The chamber's calibration matrix B (Nr x Np_eff).")

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