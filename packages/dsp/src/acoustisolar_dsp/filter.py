import numpy as np
from typing import Tuple

class DigitalBandpassFilter:
    """
    Zero-phase digital bandpass filter using frequency-domain brickwall / tapered window.
    Filters waveform to isolate the acoustic emission frequency band (e.g., 50 kHz - 400 kHz).
    """

    def __init__(self, low_cut_hz: float = 50.0e3, high_cut_hz: float = 400.0e3, sample_rate_hz: float = 1_000_000.0):
        self.low_cut = low_cut_hz
        self.high_cut = high_cut_hz
        self.sample_rate = sample_rate_hz

    def filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply zero-phase bandpass filter. Works on 1D or multichannel 2D arrays.
        """
        x = np.asarray(signal, dtype=np.float64)
        n = x.shape[-1]
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        
        # Smooth cosine-tapered frequency response to prevent Gibbs ringing
        f_safe = np.maximum(freqs, 1.0)
        hp = 1.0 / np.sqrt(1.0 + (self.low_cut / f_safe)**6)
        lp = 1.0 / np.sqrt(1.0 + (freqs / self.high_cut)**6)
        H = hp * lp
        
        if x.ndim == 1:
            return np.fft.irfft(np.fft.rfft(x) * H, n=n)
        else:
            out = np.zeros_like(x)
            for ch in range(x.shape[0]):
                out[ch] = np.fft.irfft(np.fft.rfft(x[ch]) * H, n=n)
            return out
