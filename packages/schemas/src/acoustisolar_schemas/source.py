from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class SourceClass(str, Enum):
    CRACK_INITIATION = "crack_initiation"
    CRACK_GROWTH = "crack_growth"
    CRACK_FRICTION = "crack_friction"
    NOISE = "noise"
    ACTIVE_PULSE = "active_pulse"

class CrackType(str, Enum):
    DENDRITIC = "dendritic"
    DIAGONAL = "diagonal"
    PARALLEL_TO_BUSBAR = "parallel_to_busbar"
    EDGE_CHIPPING = "edge_chipping"
    SOLDER_FRACTURE = "solder_fracture"
    NONE = "none"

class NoiseClass(str, Enum):
    RAIN = "rain"            # dense train of small drop impacts
    HAIL = "hail"            # single large ice impact (may itself crack cells)
    RAIN_HAIL = "rain_hail"  # deprecated: kept for backward compatibility, treated as HAIL
    WIND_RATTLE = "wind_rattle"
    THERMAL_TICK = "thermal_tick"
    EMI_BURST = "emi_burst"
    HANDLING_TAP = "handling_tap"
    WHITE_NOISE = "white_noise"
    NONE = "none"

class SourceEventConfig(BaseModel):
    """Acoustic emission source event specification."""
    source_class: SourceClass = Field(default=SourceClass.CRACK_INITIATION)
    crack_type: CrackType = Field(default=CrackType.DIAGONAL)
    noise_class: NoiseClass = Field(default=NoiseClass.NONE)
    x_mm: float = Field(default=520.0, ge=0.0, description="Source X coordinate in mm")
    y_mm: float = Field(default=880.0, ge=0.0, description="Source Y coordinate in mm")
    cell_id: Optional[str] = Field(default="C_10_3", description="Cell identifier containing the source")
    energy_arbitrary: float = Field(default=1.0, gt=0.0, description="Source scalar magnitude / seismic moment proxy")
    rise_time_us: float = Field(default=1.5, gt=0.0, description="Source moment-tensor rise time in microseconds")
    duration_us: float = Field(default=25.0, gt=0.0, description="Source active pulse duration in microseconds")
    orientation_deg: float = Field(default=45.0, description="Crack orientation angle relative to cell busbars in degrees")

