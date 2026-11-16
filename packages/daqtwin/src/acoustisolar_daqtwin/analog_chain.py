from typing import Optional

import numpy as np

from acoustisolar_schemas.daq import PreampConfig


class AnalogFrontEnd:
    """
    Analog front-end model:
    - preamplifier with programmable gain (dB)
    - input-referred voltage noise
    - **causal** band-pass (2nd-order Butterworth high-pass + 2nd-order low-pass, i.e. a
      4-pole band-pass), with per-channel component tolerance on the corner frequencies.

    Real analog filters are causal: they delay and smear the onset but never put energy
    before it.  The earlier zero-phase FFT mask created pre-arrival ringing that would bias
    onset pickers trained on simulated data.  ``zero_phase=True`` keeps the old behaviour
    for comparison studies only.
    """

    def __init__(
        self,
        config: Optional[PreampConfig] = None,
        seed: int = 42,
        corner_tolerance: float = 0.02,
        zero_phase: bool = False,
    ):
        self.config = config or PreampConfig()
        self.rng = np.random.default_rng(seed)
        self.linear_gain = 10.0 ** (self.config.gain_db / 20.0)
        self.corner_tolerance = corner_tolerance
        self.zero_phase = zero_phase
        self._sos_cache = {}

    def _sos(self, sample_rate_hz: float, channel: int):
        key = (sample_rate_hz, channel)
        if key not in self._sos_cache:
            from scipy.signal import butter

            rng = np.random.default_rng(1000 + channel)
            tol_lo = 1.0 + rng.normal(0.0, self.corner_tolerance)
            tol_hi = 1.0 + rng.normal(0.0, self.corner_tolerance)
            f_low = self.config.band_khz[0] * 1.0e3 * tol_lo
            f_high = min(self.config.band_khz[1] * 1.0e3 * tol_hi, 0.45 * sample_rate_hz)
            hp = butter(2, f_low, btype="highpass", fs=sample_rate_hz, output="sos")
            lp = butter(2, f_high, btype="lowpass", fs=sample_rate_hz, output="sos")
            self._sos_cache[key] = np.vstack([hp, lp])
        return self._sos_cache[key]

    def _filter(self, x: np.ndarray, sample_rate_hz: float, channel: int) -> np.ndarray:
        if self.zero_phase:
            n = x.shape[-1]
            freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
            f_low = self.config.band_khz[0] * 1.0e3
            f_high = self.config.band_khz[1] * 1.0e3
            f_safe = np.maximum(freqs, 1.0)
            H = 1.0 / np.sqrt(1.0 + (f_low / f_safe) ** 4) / np.sqrt(1.0 + (freqs / f_high) ** 4)
            return np.fft.irfft(np.fft.rfft(x) * H, n=n)
        from scipy.signal import sosfilt

        return sosfilt(self._sos(sample_rate_hz, channel), x)

    def process(
        self,
        input_voltage: np.ndarray,
        sample_rate_hz: float,
        interference_v: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Amplify and band-limit sensor voltages.

        ``interference_v`` is electrical pickup (e.g. EMI) added at the preamp input; it
        bypasses the transducer, so it is identical on every channel it couples into.
        """
        v_in = np.asarray(input_voltage, dtype=np.float64)
        squeeze = v_in.ndim == 1
        v = v_in[None, :] if squeeze else v_in
        n = v.shape[-1]
        noise_rms = self.config.input_noise_nv_rthz * 1.0e-9 * np.sqrt(sample_rate_hz / 2.0)
        out = np.zeros_like(v)
        for ch in range(v.shape[0]):
            x = v[ch] + self.rng.normal(0.0, noise_rms, size=n)
            if interference_v is not None:
                x = x + (interference_v if interference_v.ndim == 1 else interference_v[ch])
            out[ch] = self._filter(x * self.linear_gain, sample_rate_hz, ch)
        return out[0] if squeeze else out
