from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

class AntennaArray(BaseModel):
    """Defines the configuration of an antenna array."""
    num_tx_antennas: int = Field(1, gt=0, description="Number of transmitter antennas")
    num_rx_antennas: int = Field(1, gt=0, description="Number of receiver antennas")
    tx_positions: np.ndarray = Field(np.zeros((1, 3)), description="Numpy array of transmitter antenna positions")
    rx_positions: np.ndarray = Field(np.zeros((1, 3)), description="Numpy array of receiver antenna positions")
    
    # Spatial correlation parameters
    tx_array_type: str = Field('ULA', description="TX array type, e.g., 'ULA' for Uniform Linear Array")
    rx_array_type: str = Field('ULA', description="RX array type, e.g., 'ULA' for Uniform Linear Array")
    tx_element_spacing: float = Field(0.5, description="TX element spacing in multiples of wavelength")
    rx_element_spacing: float = Field(0.5, description="RX element spacing in multiples of wavelength")

    # Element pattern parameters
    tx_element_pattern: str = Field('Isotropic', description="TX element pattern, e.g., 'Isotropic' or 'TR38.901'")
    rx_element_pattern: str = Field('Isotropic', description="RX element pattern, e.g., 'Isotropic' or 'TR38.901'")

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
    antenna_config: AntennaArray = Field(AntennaArray(), description="Antenna array configuration")

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