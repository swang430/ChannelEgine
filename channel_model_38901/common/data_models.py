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
    # Array orientation in the GCS (3GPP TR 38.901 Sec 7.1.3). All 0 => identity.
    tx_bearing_deg: float = 0.0
    tx_downtilt_deg: float = 0.0
    tx_slant_deg: float = 0.0
    tx_positions: Optional[np.ndarray] = None

    rx_array_type: str = 'ULA'
    num_rx_rows: int = 1
    num_rx_cols: int = 1
    rx_spacing_h: float = 0.5
    rx_spacing_v: float = 0.5
    rx_element_pattern: str = 'Isotropic'
    rx_bearing_deg: float = 0.0
    rx_downtilt_deg: float = 0.0
    rx_slant_deg: float = 0.0
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
    # zenith 约定 (θ 从 +z 量起), 默认 90° = 水平方向, 与 mimo_ota_simulator
    # 一致 (audit batch 2, A2; 历史默认 0.0 在 zenith 约定下意为"垂直向上")
    zod_theta: float = 90.0
    zoa_theta: float = 90.0
    as_aod: float = 0.0
    as_aoa: float = 0.0
    as_zod: float = 0.0
    as_zoa: float = 0.0
    k_n: Optional[float] = 1.0
    initial_phases: Optional[np.ndarray] = None
    h_matrix: Optional[np.ndarray] = None
    # Phase 5 plumb-only field (see docs/plans/external-cdl-input-design.md §4).
    # Stored when CustomCDLProfile.clusters[i].xpr_db is provided; converted to linear.
    # Synthesizer does NOT consume this field in Phase 5; semantic application
    # (dual-pol synthesizer) lands in Phase 6.
    xpr_linear: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

class ChannelProfile(BaseModel):
    """A standardized, internal representation of a channel model's parameters."""
    name: str
    clusters: List[EngineCluster]
    num_tx_antennas: int
    num_rx_antennas: int