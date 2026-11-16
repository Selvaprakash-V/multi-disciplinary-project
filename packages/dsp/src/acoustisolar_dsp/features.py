"""
Acoustic-emission feature extraction.

Per-channel (classic AE hit parameters + spectral + modal):
    peak amplitude, rise time, duration, counts, energy, spectral centroid, peak frequency,
    RA value (rise time / amplitude), average frequency (counts / duration),
    partial power in 4 bands, early/late energy ratio around the first arrival
    (a proxy for the S0/A0 content: cracks are relatively S0-rich, impacts A0-rich).

Event-level (multichannel / context):
    arrival spread (~0 for electrical EMI, which is not acoustic), number of channels
    above threshold, max/min channel amplitude ratio, distance of the located source to the
    nearest frame edge (frame stick-slip and rattle occur at the edge), temperature.

Channel features are ordered by **arrival order** (earliest first) rather than sensor index
so that a model does not learn position-specific patterns tied to sensor numbering.
"""

from typing import Dict, List, Optional

import numpy as np

from acoustisolar_schemas.event import TriggerFeatures

from .pickers import OnsetPicker

PARTIAL_BANDS_KHZ = ((50.0, 100.0), (100.0, 150.0), (150.0, 250.0), (250.0, 400.0))
CHANNEL_FEATURES = [
    "peak_amp_db",
    "rise_time_us",
    "duration_us",
    "counts",
    "log_energy",
    "centroid_khz",
    "peak_freq_khz",
    "ra_us_per_mv",
    "avg_freq_khz",
    "pp_50_100",
    "pp_100_150",
    "pp_150_250",
    "pp_250_400",
    "early_late_ratio_db",
]
EVENT_FEATURES = [
    "arrival_spread_us",
    "n_channels_hit",
    "amp_ratio_db",
    "edge_distance_mm",
    "temperature_c",
]


class AEFeatureExtractor:
    def __init__(self, sample_rate_hz: float = 1_000_000.0, threshold_volts: float = 0.020):
        self.sample_rate = sample_rate_hz
        self.dt_us = 1.0e6 / sample_rate_hz
        self.threshold_v = threshold_volts

    # ------------------------------------------------------------------ single channel
    def extract_features(self, waveform_volts: np.ndarray) -> TriggerFeatures:
        """Classic + extended AE features for one channel."""
        v = np.asarray(waveform_volts, dtype=np.float64)
        n = len(v)
        abs_v = np.abs(v)
        peak_idx = int(np.argmax(abs_v))
        peak_amp_mv = float(abs_v[peak_idx] * 1000.0)

        above = np.where(abs_v >= self.threshold_v)[0]
        if len(above) == 0:
            first, last, counts = peak_idx, peak_idx, 0
            rise_time_us = 0.0
            duration_us = 0.0
        else:
            first, last = int(above[0]), int(above[-1])
            counts = int(np.sum(np.diff((v >= self.threshold_v).astype(int)) > 0))
            rise_time_us = float(max(0, peak_idx - first) * self.dt_us)
            duration_us = float(max(0, last - first) * self.dt_us)

        energy_au = float(np.sum((v * 1000.0) ** 2) * self.dt_us)

        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spec = np.abs(np.fft.rfft(v))
        power = spec**2
        total = float(np.sum(spec))
        if total > 1e-12:
            centroid_hz = float(np.sum(freqs * spec) / total)
            peak_freq_hz = float(freqs[np.argmax(spec)])
        else:
            centroid_hz = peak_freq_hz = 0.0
        p_tot = float(np.sum(power)) + 1e-30
        partial = [
            float(np.sum(power[(freqs >= lo * 1e3) & (freqs < hi * 1e3)]) / p_tot) for lo, hi in PARTIAL_BANDS_KHZ
        ]

        # early/late energy around the first arrival (modal-content proxy)
        onset = OnsetPicker.threshold_onset(v, k_sigma=6.0)
        if onset < 0:
            onset = first
        e_early = float(np.sum(v[onset : onset + 25] ** 2)) + 1e-30
        e_late = float(np.sum(v[onset + 25 : onset + 250] ** 2)) + 1e-30

        return TriggerFeatures(
            peak_amp_mv=peak_amp_mv,
            rise_time_us=rise_time_us,
            duration_us=duration_us,
            counts=counts,
            energy_au=energy_au,
            centroid_khz=centroid_hz * 1.0e-3,
            peak_freq_khz=peak_freq_hz * 1.0e-3,
            ra_us_per_mv=rise_time_us / max(peak_amp_mv, 1e-6),
            avg_freq_khz=(counts / duration_us * 1.0e3) if duration_us > 0 else 0.0,
            partial_powers=partial,
            early_late_ratio_db=10.0 * np.log10(e_early / e_late),
            onset_sample=int(onset),
        )

    @staticmethod
    def _channel_vector(f: TriggerFeatures) -> List[float]:
        return [
            20.0 * np.log10(max(f.peak_amp_mv, 1e-3)),
            f.rise_time_us,
            f.duration_us,
            float(f.counts),
            float(np.log10(f.energy_au + 1e-6)),
            f.centroid_khz,
            f.peak_freq_khz,
            f.ra_us_per_mv,
            f.avg_freq_khz,
            *f.partial_powers,
            f.early_late_ratio_db,
        ]

    # ------------------------------------------------------------------ multichannel
    def extract_event_features(
        self,
        multichannel_volts: np.ndarray,
        location_mm: Optional[tuple] = None,
        module_size_mm: Optional[tuple] = None,
        temperature_c: float = 25.0,
    ) -> Dict[str, object]:
        """Per-channel features (arrival-ordered) + event-level features."""
        feats = [self.extract_features(ch) for ch in multichannel_volts]
        onsets = np.array([f.onset_sample for f in feats], dtype=float)
        order = np.argsort(onsets, kind="stable")
        amps = np.array([f.peak_amp_mv for f in feats])
        hit = amps >= self.threshold_v * 1000.0
        valid_onsets = onsets[hit] if hit.sum() >= 2 else onsets
        edge = -1.0
        if location_mm is not None and module_size_mm is not None:
            x, y = location_mm
            W, L = module_size_mm
            edge = float(max(0.0, min(x, W - x, y, L - y)))
        event = {
            "arrival_spread_us": float((np.max(valid_onsets) - np.min(valid_onsets)) * self.dt_us),
            "n_channels_hit": float(hit.sum()),
            "amp_ratio_db": float(20.0 * np.log10(max(amps.max(), 1e-6) / max(amps.min(), 1e-6))),
            "edge_distance_mm": edge,
            "temperature_c": float(temperature_c),
        }
        return {"channels": [feats[i] for i in order], "event": event, "arrival_order": order.tolist()}

    def extract_channel_features(
        self,
        multichannel_volts: np.ndarray,
        location_mm: Optional[tuple] = None,
        module_size_mm: Optional[tuple] = None,
        temperature_c: float = 25.0,
    ) -> np.ndarray:
        """Flat feature vector: arrival-ordered channel features + event features."""
        ev = self.extract_event_features(multichannel_volts, location_mm, module_size_mm, temperature_c)
        vec: List[float] = []
        for f in ev["channels"]:
            vec.extend(self._channel_vector(f))
        vec.extend(ev["event"][k] for k in EVENT_FEATURES)
        return np.array(vec, dtype=np.float32)

    @staticmethod
    def feature_names(n_channels: int = 4) -> List[str]:
        names = [f"ch{r}_{n}" for r in range(n_channels) for n in CHANNEL_FEATURES]
        return names + list(EVENT_FEATURES)
