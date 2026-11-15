import numpy as np
from typing import Optional
from acoustisolar_schemas.daq import ADCConfig

class ADCTwin:
    """
    Analog-to-Digital Converter twin:
    - Quantization to N bits
    - Full-scale clipping / saturation
    - Clock jitter noise
    - Inter-channel sampling skew
    """

    def __init__(self, config: Optional[ADCConfig] = None, seed: int = 42):
        self.config = config or ADCConfig()
        self.rng = np.random.default_rng(seed)
        self.max_counts = (1 << self.config.bits) - 1
        self.v_fs = self.config.full_scale_volts
        self.lsb_volts = self.v_fs / self.max_counts

    def digitize(self, analog_volts: np.ndarray, channel_skew_ns: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert continuous voltage array to discrete integer ADC counts (int16).
        Simulates channel skew and sampling jitter.
        """
        v = np.asarray(analog_volts, dtype=np.float64)
        n_samples = v.shape[-1]
        
        # Center voltage signal around ADC midpoint (V_fs / 2)
        v_offset = v + (self.v_fs / 2.0)
        
        # Apply inter-channel skew via sub-sample linear interpolation if multichannel
        if v.ndim == 2 and channel_skew_ns is not None:
            dt_s = 1.0 / self.config.rate_hz
            for ch in range(v.shape[0]):
                shift_samples = (channel_skew_ns[ch] * 1.0e-9) / dt_s
                if abs(shift_samples) > 1e-6:
                    indices = np.arange(n_samples) - shift_samples
                    v_offset[ch] = np.interp(indices, np.arange(n_samples), v_offset[ch], left=v_offset[ch, 0], right=v_offset[ch, -1])
                    
        # Apply clock sampling jitter
        if self.config.jitter_ns > 0:
            jitter_frac = (self.config.jitter_ns * 1.0e-9) * self.config.rate_hz
            jitter_noise = self.rng.normal(0.0, jitter_frac, size=v.shape)
            # Jitter modulates voltage proportionally to signal slope dv/dt
            dv = np.gradient(v_offset, axis=-1)
            v_offset += dv * jitter_noise

        # Quantize and clip to [0, max_counts]
        counts = np.round(v_offset / self.lsb_volts)
        counts = np.clip(counts, 0, self.max_counts).astype(np.int16)
        return counts

    def counts_to_volts(self, counts: np.ndarray) -> np.ndarray:
        """Convert raw ADC counts back to calibrated input voltage (V)."""
        counts = np.asarray(counts, dtype=np.float64)
        return (counts * self.lsb_volts) - (self.v_fs / 2.0)

