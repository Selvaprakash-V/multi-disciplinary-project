import numpy as np
from typing import Dict, Any
from acoustisolar_schemas.event import TriggerFeatures

class AEFeatureExtractor:
    """
    Classic acoustic emission waveform feature extraction:
    - Peak amplitude (mV)
    - Rise time (us)
    - Duration (us)
    - Ringdown counts
    - Absolute energy
    - Spectral centroid (kHz)
    - Partial power frequency bands
    """

    def __init__(self, sample_rate_hz: float = 1_000_000.0, threshold_volts: float = 0.020):
        self.sample_rate = sample_rate_hz
        self.dt_us = (1.0 / sample_rate_hz) * 1.0e6
        self.threshold_v = threshold_volts

    def extract_features(self, waveform_volts: np.ndarray) -> TriggerFeatures:
        """Extract AE features from a single channel voltage waveform."""
        v = np.asarray(waveform_volts, dtype=np.float64)
        n = len(v)
        abs_v = np.abs(v)
        
        # Peak amplitude
        peak_idx = int(np.argmax(abs_v))
        peak_amp_mv = float(abs_v[peak_idx] * 1000.0)
        
        # Threshold crossings
        above_thresh = np.where(abs_v >= self.threshold_v)[0]
        
        if len(above_thresh) == 0:
            first_cross = peak_idx
            last_cross = peak_idx
            counts = 0
            rise_time_us = float(peak_idx * self.dt_us)
            duration_us = 0.0
        else:
            first_cross = int(above_thresh[0])
            last_cross = int(above_thresh[-1])
            # Count zero-crossings or positive-going threshold crossings
            pos_cross = np.diff((v >= self.threshold_v).astype(int))
            counts = int(np.sum(pos_cross > 0))
            rise_samples = max(0, peak_idx - first_cross)
            rise_time_us = float(rise_samples * self.dt_us)
            duration_us = float(max(0, last_cross - first_cross) * self.dt_us)
            
        # Absolute Energy: integral of V^2 dt (in mV^2 * us)
        energy_au = float(np.sum((v * 1000.0)**2) * self.dt_us)
        
        # Spectral features via FFT
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spec = np.abs(np.fft.rfft(v))
        total_spec = np.sum(spec)
        
        if total_spec > 1e-12:
            centroid_hz = float(np.sum(freqs * spec) / total_spec)
            peak_freq_hz = float(freqs[np.argmax(spec)])
        else:
            centroid_hz = 0.0
            peak_freq_hz = 0.0
            
        return TriggerFeatures(
            peak_amp_mv=peak_amp_mv,
            rise_time_us=rise_time_us,
            duration_us=duration_us,
            counts=counts,
            energy_au=energy_au,
            centroid_khz=centroid_hz * 1.0e-3,
            peak_freq_khz=peak_freq_hz * 1.0e-3,
        )

    def extract_channel_features(self, multichannel_volts: np.ndarray) -> np.ndarray:
        """
        Extract feature vector for multichannel array (shape: n_channels, n_samples).
        Returns 1D feature array suitable for ML classifiers.
        """
        feats = []
        for ch in range(multichannel_volts.shape[0]):
            f = self.extract_features(multichannel_volts[ch])
            feats.extend([
                f.peak_amp_mv,
                f.rise_time_us,
                f.duration_us,
                float(f.counts),
                f.energy_au,
                f.centroid_khz,
                f.peak_freq_khz,
            ])
        return np.array(feats, dtype=np.float32)
