
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

class AntennaArray(BaseModel):
    # Using BaseModel for nested structures
    pass

class ScenarioConfig(BaseModel):
    """
    Holds user-defined simulation configuration.
    """
    scenario_name: str = Field(..., description="Name of the scenario, e.g., 'UMa', 'UMi-StreetCanyon'")
    center_frequency_hz: float = Field(..., gt=0, description="Center frequency in Hz")
    bs_positions: np.ndarray = Field(..., description="Numpy array of base station positions")
    ue_positions: np.ndarray = Field(..., description="Numpy array of user equipment positions")
    ue_velocities: np.ndarray = Field(..., description="Numpy array of user equipment velocities")
    antenna_config: Optional[AntennaArray] = Field(None, description="Antenna array configuration")

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

