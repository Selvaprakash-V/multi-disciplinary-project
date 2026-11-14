import numpy as np
from typing import Optional
from acoustisolar_schemas.sensor import PZTDiscConfig

class PZTSensorModel:
    """
    Piezoelectric PZT disc acoustic sensor model.
    Models frequency-dependent resonant transduction, mechanical damping Q,
    and circular aperture directivity.
    """

    def __init__(self, config: Optional[PZTDiscConfig] = None, seed: int = 42):
        self.config = config or PZTDiscConfig()
        self.rng = np.random.default_rng(seed)
        
        # Apply manufacturing sensitivity spread (dB)
        spread_db = self.rng.normal(0.0, self.config.sensitivity_spread_db / 2.0)
        self.calibrated_gain = 10.0 ** (spread_db / 20.0)

    def transfer_function(self, freqs_hz: np.ndarray) -> np.ndarray:
        """
        Compute complex frequency response H_pzt(f) of the resonant PZT transducer.
        Second-order resonant system with center frequency f_0 and quality factor Q.
        """
        f = np.asarray(freqs_hz, dtype=np.float64)
        f0 = self.config.resonance_khz * 1.0e3
        Q = self.config.quality_factor
        
        omega = 2.0 * np.pi * f
        omega0 = 2.0 * np.pi * f0
        gamma = omega0 / Q
        
        # Resonant bandpass transfer function
        numerator = 1j * omega * gamma
        denominator = (omega0**2 - omega**2) + 1j * omega * gamma
        
        # Guard against omega=0 division
        H = np.where(omega > 0, numerator / denominator, 0.0)
        return H * self.config.sensitivity_v_per_m_s * self.calibrated_gain

    def apply_transduction(self, surface_velocity: np.ndarray, sample_rate_hz: float) -> np.ndarray:
        """
        Convert surface mechanical velocity wave (m/s) to sensor output voltage (V).
        """
        n = surface_velocity.shape[-1]
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
        H = self.transfer_function(freqs)
        
        if surface_velocity.ndim == 1:
            V_fft = np.fft.rfft(surface_velocity) * H
            return np.fft.irfft(V_fft, n=n)
        else:
            V_out = np.zeros_like(surface_velocity)
            for ch in range(surface_velocity.shape[0]):
                V_fft = np.fft.rfft(surface_velocity[ch]) * H
                V_out[ch] = np.fft.irfft(V_fft, n=n)
            return V_out
