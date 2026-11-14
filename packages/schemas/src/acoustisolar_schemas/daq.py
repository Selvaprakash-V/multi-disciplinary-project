from pydantic import BaseModel, Field
from typing import Tuple

class PreampConfig(BaseModel):
    """Analog preamplifier configuration."""
    gain_db: float = Field(default=40.0, description="Preamplifier voltage gain in dB")
    input_noise_nv_rthz: float = Field(default=5.0, ge=0.0, description="Input-referred voltage noise density in nV/sqrt(Hz)")
    band_khz: Tuple[float, float] = Field(default=(50.0, 400.0), description="Analog passband (low_cut_khz, high_cut_khz)")
    input_impedance_kohm: float = Field(default=100.0, gt=0, description="Input impedance in kilo-ohms")

class ADCConfig(BaseModel):
    """Analog-to-Digital Converter digital twin parameters."""
    bits: int = Field(default=12, ge=8, le=24, description="ADC bit resolution")
    rate_hz: float = Field(default=1_000_000.0, gt=0, description="Simultaneous sampling rate in Hz")
    full_scale_volts: float = Field(default=3.3, gt=0, description="Full scale input voltage range in Volts")
    jitter_ns: float = Field(default=20.0, ge=0.0, description="Clock jitter standard deviation in nanoseconds")
    skew_ns: float = Field(default=50.0, ge=0.0, description="Max inter-channel sampling skew in nanoseconds")

class TriggerConfig(BaseModel):
    """Hardware trigger and event windowing logic."""
    threshold_mv: float = Field(default=20.0, gt=0, description="Analog trigger threshold in millivolts")
    pre_samples: int = Field(default=512, ge=0, description="Pre-trigger buffer length in samples")
    post_samples: int = Field(default=2048, gt=0, description="Post-trigger capture length in samples")
    dead_time_us: float = Field(default=500.0, ge=0.0, description="Hardware trigger hold-off / dead time in microseconds")

    @property
    def total_samples(self) -> int:
        return self.pre_samples + self.post_samples

class DAQConfig(BaseModel):
    """Complete DAQ twin configuration."""
    preamp: PreampConfig = Field(default_factory=PreampConfig)
    adc: ADCConfig = Field(default_factory=ADCConfig)
    trigger: TriggerConfig = Field(default_factory=TriggerConfig)
