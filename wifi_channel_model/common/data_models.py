from typing import List, Optional, Literal
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class WiFiScenarioConfig(BaseModel):
    """Configuration for WiFi channel generation (TGax-based)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(..., description="WiFi channel model name, e.g., TGax-D.")
    center_frequency_hz: float = Field(..., description="Center frequency in Hz.")
    bandwidth_hz: float = Field(..., description="Channel bandwidth in Hz.")
    enable_ofdm: bool = Field(False, description="Enable OFDM subcarrier H computation.")
    n_fft: Optional[int] = Field(None, description="FFT size for OFDM (optional).")
    sample_rate_hz: Optional[float] = Field(None, description="Sample rate for impulse response (optional).")
    random_seed: Optional[int] = Field(None, description="Random seed for deterministic tap phases.")
    phase_policy: Literal["zero", "random"] = Field("zero", description="Tap phase policy.")
    normalize_power: bool = Field(True, description="Normalize tap powers to sum to 1.")
    delay_scale: Optional[float] = Field(None, description="Manual delay scaling factor for tap delays.")

    bs_positions: np.ndarray = Field(
        default_factory=lambda: np.array([[0.0, 0.0, 1.5]]),
        description="Base station positions (m)."
    )
    ue_positions: np.ndarray = Field(
        default_factory=lambda: np.array([[1.0, 0.0, 1.5]]),
        description="User equipment positions (m)."
    )
    ue_velocities: np.ndarray = Field(
        default_factory=lambda: np.array([[0.0, 0.0, 0.0]]),
        description="User equipment velocities (m/s)."
    )


class WiFiTap(BaseModel):
    """Tap-level delay and power."""
    delay_s: float = 0.0
    power_linear: float = 0.0
    cluster_id: int = 0


class WiFiCluster(BaseModel):
    """Cluster-level angle statistics."""
    cluster_id: int = 0
    aoa_deg: float = 0.0
    aod_deg: float = 0.0
    as_rx_deg: float = 0.0
    as_tx_deg: float = 0.0


class TapComponent(BaseModel):
    """Tap component with inherited angle statistics."""
    delay_s: float = 0.0
    power_linear: float = 0.0
    aoa_deg: float = 0.0
    aod_deg: float = 0.0
    as_rx_deg: float = 0.0
    as_tx_deg: float = 0.0


class WiFiChannelRealization(BaseModel):
    """Unified WiFi channel realization output."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    channel_matrix: Optional[np.ndarray] = None
    impulse_response: Optional[List[dict]] = None
    taps: Optional[List[WiFiTap]] = None
    clusters: Optional[List[WiFiCluster]] = None
    tap_components: Optional[List[TapComponent]] = None
    h_subcarriers: Optional[np.ndarray] = None
