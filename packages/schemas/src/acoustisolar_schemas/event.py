import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional
from .source import CrackType, NoiseClass, SourceClass

class TriggerFeatures(BaseModel):
    """Classic acoustic emission waveform features computed on trigger."""
    peak_amp_mv: float = Field(..., description="Peak signal amplitude in millivolts")
    rise_time_us: float = Field(..., description="Time from first threshold crossing to peak in microseconds")
    duration_us: float = Field(..., description="Time from first to last threshold crossing in microseconds")
    counts: int = Field(..., ge=0, description="Number of threshold crossings (ringdown counts)")
    energy_au: float = Field(..., ge=0.0, description="Integrated squared voltage (AE energy)")
    centroid_khz: float = Field(..., ge=0.0, description="Spectral centroid frequency in kHz")
    peak_freq_khz: float = Field(default=0.0, ge=0.0, description="Peak spectral frequency in kHz")

class GroundTruthLabel(BaseModel):
    """Ground-truth metadata for synthetic or verified lab events."""
    source_class: SourceClass
    crack_type: CrackType = CrackType.NONE
    noise_class: NoiseClass = NoiseClass.NONE
    x_mm: float
    y_mm: float
    cell_id: Optional[str] = None
    energy: float = 1.0
    severity: float = Field(default=1.0, ge=0.0, description="Crack severity score")

class ProvenanceRecord(BaseModel):
    """Provenance tracking for dataset reproducibility."""
    sim_tier: Literal["T0", "T1", "T2", "REAL"] = "T0"
    config_hash: str = Field(..., description="SHA-256 hash of configuration")
    code_sha: str = Field(default="dev", description="Git commit SHA")
    seed: int = Field(default=42, description="Random seed used for generation")
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class EventRecord(BaseModel):
    """Complete waveform event record matching PRD section 8.2 and Appendix B."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    module_id: str = Field(default="MOD-DEMO-001")
    node_id: str = Field(default="NODE-001")
    t_trigger_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sample_rate_hz: float = Field(default=1_000_000.0)
    n_channels: int = Field(default=4)
    pre_trigger_samples: int = Field(default=512)
    total_samples: int = Field(default=2560)
    channel_skew_ns: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    mode: Literal["P", "A", "H"] = Field(default="P", description="P=Passive AE, A=Active ultrasonic, H=Hybrid")
    temperature_c: Optional[float] = 25.0
    irradiance_wm2: Optional[float] = 800.0
    wind_ms: Optional[float] = 1.5
    rain_flag: bool = False
    trigger_features: TriggerFeatures
    label: Optional[GroundTruthLabel] = None
    provenance: ProvenanceRecord

