from pydantic import BaseModel, Field
from typing import List, Tuple

class PZTDiscConfig(BaseModel):
    """Specification of the piezoelectric sensor disc."""
    diameter_mm: float = Field(default=10.0, gt=0, description="PZT disc diameter in mm")
    thickness_mm: float = Field(default=1.0, gt=0, description="PZT disc thickness in mm")
    resonance_khz: float = Field(default=200.0, gt=0, description="Radial/thickness resonant frequency in kHz")
    quality_factor: float = Field(default=5.0, gt=0, description="Mechanical resonance quality factor Q")
    coupling_factor_keff: float = Field(default=0.35, ge=0.0, le=1.0, description="Effective electromechanical coupling factor")
    capacitance_pf: float = Field(default=1200.0, gt=0, description="Static sensor capacitance in pF")
    sensitivity_v_per_m_s: float = Field(default=15.0, gt=0, description="Voltage output sensitivity V/(m/s)")
    sensitivity_spread_db: float = Field(default=3.0, ge=0.0, description="Manufacturing sensitivity variation in dB")

class SensorArrayConfig(BaseModel):
    """Array configuration of acoustic emission sensors on the PV module."""
    count: int = Field(default=4, ge=2, description="Number of AE sensors in the array")
    positions_mm: List[Tuple[float, float]] = Field(
        default_factory=lambda: [
            (60.0, 60.0),
            (60.0, 1640.0),
            (980.0, 60.0),
            (980.0, 1640.0),
        ],
        description="List of (x, y) coordinates for each sensor in millimeters"
    )
    disc: PZTDiscConfig = Field(default_factory=PZTDiscConfig)

