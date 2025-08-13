from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

class AntennaArray(BaseModel):
    """Configuration of the antenna array."""
    tx_array_type: str = 'ULA'
    num_tx_rows: int = 1
    num_tx_cols: int = 1
    tx_spacing_h: float = 0.5
    tx_spacing_v: float = 0.5
    tx_element_pattern: str = 'Isotropic'
    tx_positions: Optional[np.ndarray] = None

    rx_array_type: str = 'ULA'
    num_rx_rows: int = 1
    num_rx_cols: int = 1
    rx_spacing_h: float = 0.5
    rx_spacing_v: float = 0.5
    rx_element_pattern: str = 'Isotropic'
    rx_positions: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

class ScenarioConfig(BaseModel):
    """Holds all user-defined simulation parameters."""
    scenario_name: str
    center_frequency_hz: float
    cluster_model_name: Optional[str] = None
    bs_positions: np.ndarray
    ue_positions: np.ndarray
    ue_velocities: np.ndarray
    antenna_config: AntennaArray = Field(default_factory=AntennaArray)

    class Config:
        arbitrary_types_allowed = True

class LargeScaleParameters(BaseModel):
    """Encapsulates a set of large-scale fading parameters."""
    delay_spread_s: float
    departure_azimuth_spread_deg: float
    arrival_azimuth_spread_deg: float
    departure_zenith_spread_deg: float
    arrival_zenith_spread_deg: float
    k_factor_linear: float
    shadow_fade_db: float

class EngineCluster(BaseModel):
    """A detailed, unified definition for a channel cluster."""
    delay: float = 0.0
    power: float = 0.0
    aod_phi: float = 0.0
    aoa_phi: float = 0.0
    zod_theta: float = 0.0
    zoa_theta: float = 0.0
    k_n: Optional[float] = 1.0
    initial_phases: Optional[np.ndarray] = None
    h_matrix: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

class ChannelProfile(BaseModel):
    """A standardized, internal representation of a channel model's parameters."""
    name: str
    clusters: List[EngineCluster]
    num_tx_antennas: int
    num_rx_antennas: int