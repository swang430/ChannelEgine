from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Literal
import numpy as np

class AntennaArray(BaseModel):
    """Defines the configuration of an antenna array."""
    # --- TX Array ---
    tx_array_type: Literal['ULA', 'URA'] = Field('ULA', description="TX array type: Uniform Linear or Uniform Rectangular Array")
    num_tx_rows: int = Field(1, gt=0, description="Number of rows in the TX antenna array")
    num_tx_cols: int = Field(1, gt=0, description="Number of columns in the TX antenna array (for ULA, rows=1)")
    tx_spacing_h: float = Field(0.5, description="Horizontal spacing between TX elements in multiples of wavelength")
    tx_spacing_v: float = Field(0.5, description="Vertical spacing between TX elements in multiples of wavelength (for URA)")
    tx_element_pattern: str = Field('Isotropic', description="TX element pattern, e.g., 'Isotropic' or 'TR38.901'")
    tx_positions: np.ndarray = Field(default_factory=lambda: np.zeros((1, 3)), description="Numpy array of transmitter antenna positions")

    # --- RX Array ---
    rx_array_type: Literal['ULA', 'URA'] = Field('ULA', description="RX array type: Uniform Linear or Uniform Rectangular Array")
    num_rx_rows: int = Field(1, gt=0, description="Number of rows in the RX antenna array")
    num_rx_cols: int = Field(1, gt=0, description="Number of columns in the RX antenna array (for ULA, rows=1)")
    rx_spacing_h: float = Field(0.5, description="Horizontal spacing between RX elements in multiples of wavelength")
    rx_spacing_v: float = Field(0.5, description="Vertical spacing between RX elements in multiples of wavelength (for URA)")
    rx_element_pattern: str = Field('Isotropic', description="RX element pattern, e.g., 'Isotropic' or 'TR38.901'")
    rx_positions: np.ndarray = Field(default_factory=lambda: np.zeros((1, 3)), description="Numpy array of receiver antenna positions")

    class Config:
        arbitrary_types_allowed = True

class ScenarioConfig(BaseModel):
    """
    Holds user-defined simulation configuration.
    """
    scenario_name: str = Field(..., description="Name of the scenario, e.g., 'UMa', 'UMi-StreetCanyon'")
    center_frequency_hz: float = Field(..., gt=0, description="Center frequency in Hz")
    bs_positions: np.ndarray = Field(..., description="Numpy array of base station positions")
    ue_positions: np.ndarray = Field(..., description="Numpy array of user equipment positions")
    ue_velocities: np.ndarray = Field(..., description="Numpy array of user equipment velocities")
    antenna_config: AntennaArray = Field(default_factory=AntennaArray, description="Antenna array configuration")

    class Config:
        arbitrary_types_allowed = True

class LargeScaleParameters(BaseModel):
    """
    Encapsulates a set of large-scale parameters for a channel realization.
    """
    delay_spread_s: float = Field(..., description="Delay Spread in seconds")
    departure_azimuth_spread_deg: float = Field(..., description="Departure Azimuth Spread in degrees")
    arrival_azimuth_spread_deg: float = Field(..., description="Arrival Azimuth Spread in degrees")
    departure_zenith_spread_deg: float = Field(..., description="Departure Zenith Spread in degrees")
    arrival_zenith_spread_deg: float = Field(..., description="Arrival Zenith Spread in degrees")
    k_factor_linear: float = Field(..., description="K-factor in linear scale")
    shadow_fade_db: float = Field(..., description="Shadow Fading in dB")
