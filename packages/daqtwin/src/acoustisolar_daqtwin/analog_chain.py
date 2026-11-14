import numpy as np
from typing import Optional
from acoustisolar_schemas.daq import PreampConfig

class AnalogFrontEnd:
    """
    Analog front-end modeling:
    - Preamplifier with programmable gain (dB)
    - Input-referred thermal and semiconductor voltage noise
    - Analog bandpass anti-aliasing filter
    """

    def __init__(self, config: Optional[PreampConfig] = None, seed: int = 42):
        self.config = config or PreampConfig()
        self.rng = np.random.default_rng(seed)
        self.linear_gain = 10.0 ** (self.config.gain_db / 20.0)

    def process(self, input_voltage: np.ndarray, sample_rate_hz: float) -> np.ndarray:
        """
        Process voltage signal through preamp and analog bandpass filter.
        """
        v_in = np.asarray(input_voltage, dtype=np.float64)
        n = v_in.shape[-1]
        dt = 1.0 / sample_rate_hz
        freqs = np.fft.rfftfreq(n, d=dt)
        
        # 1. Analog bandpass filter transfer function (4th-order Butterworth approximation)
        f_low = self.config.band_khz[0] * 1.0e3
        f_high = self.config.band_khz[1] * 1.0e3
        
        # Guard against f=0 in highpass portion
        f_safe = np.maximum(freqs, 1.0)
        hp = 1.0 / np.sqrt(1.0 + (f_low / f_safe)**4)
        lp = 1.0 / np.sqrt(1.0 + (freqs / f_high)**4)
        H_filter = hp * lp
        
        # 2. Synthesize input-referred noise: e_n [V/sqrt(Hz)] * sqrt(Nyquist bandwidth)
        bandwidth = sample_rate_hz / 2.0
        e_n = self.config.input_noise_nv_rthz * 1.0e-9
        noise_rms = e_n * np.sqrt(bandwidth)
        
        if v_in.ndim == 1:
            noise = self.rng.normal(0.0, noise_rms, size=n)
            v_total = (v_in + noise) * self.linear_gain
            V_fft = np.fft.rfft(v_total) * H_filter
            return np.fft.irfft(V_fft, n=n)
        else:
            v_out = np.zeros_like(v_in)
            for ch in range(v_in.shape[0]):
                noise = self.rng.normal(0.0, noise_rms, size=n)
                v_total = (v_in[ch] + noise) * self.linear_gain
                V_fft = np.fft.rfft(v_total) * H_filter
                v_out[ch] = np.fft.irfft(V_fft, n=n)
            return v_out
