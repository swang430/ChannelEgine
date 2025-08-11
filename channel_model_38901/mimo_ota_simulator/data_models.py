from pydantic import BaseModel, Field
from typing import List, Dict, Literal
import numpy as np

# This model now correctly imports the unified EngineCluster and ChannelProfile
# from the common data models file, so they are not redefined here.
from ..common.data_models import EngineCluster, ChannelProfile

class ChamberConfig(BaseModel):
    """Configuration of the physical OTA chamber."""
    num_probes: int = Field(..., description="Total number of probe antennas.")
    distribution: Literal['ring', 'multi-ring'] = Field(..., description="Layout of the probes.")
    radius_m: float = Field(..., description="Radius of the probe ring in meters.")
    probe_locations: Dict[int, np.ndarray] = Field({}, description="3D coordinates of each probe, key is probe ID.")
    calibration_matrix_B: np.ndarray = Field(None, description="Chamber transfer matrix (Nr x Np), from probes to DUT antennas.")

    class Config:
        arbitrary_types_allowed = True

class TargetChannelConfig(BaseModel):
    """Configuration of the target channel to be emulated."""
    model_name: str = Field(..., description="Name of the channel model (e.g., 'CDL-A', 'UMa-NLOS').")
    mimo_config: str = Field(..., description="MIMO configuration (e.g., '2x2', '4x2').")
    ue_speed_kph: float = Field(0.0, description="UE speed in km/h.")