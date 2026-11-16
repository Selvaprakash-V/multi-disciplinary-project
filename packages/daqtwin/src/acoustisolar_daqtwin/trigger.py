from typing import List, Optional, Tuple

import numpy as np

from acoustisolar_schemas.daq import TriggerConfig


class TriggerEngine:
    """
    Hardware/firmware event trigger:
    - fixed amplitude threshold (mV), or a **floating threshold** = k x running noise RMS
      (keeps the false-trigger rate bounded when rain or wind raise the noise floor)
    - pre/post-trigger window extraction
    - multi-hit detection in a continuous record with hit lockout (dead time), so that
      overlapping events (e.g. a hail impact followed by a crack burst) are separated
    """

    def __init__(
        self,
        config: Optional[TriggerConfig] = None,
        sample_rate_hz: float = 1_000_000.0,
        floating_k: Optional[float] = None,
        noise_window: int = 256,
    ):
        self.config = config or TriggerConfig()
        self.sample_rate = sample_rate_hz
        self.dead_samples = int((self.config.dead_time_us * 1.0e-6) * sample_rate_hz)
        self.floating_k = floating_k
        self.noise_window = noise_window

    # ------------------------------------------------------------------ thresholds
    def threshold_volts(self, x: np.ndarray) -> np.ndarray:
        """Per-sample threshold (V): fixed, or max(fixed, k x causal running RMS)."""
        fixed = self.config.threshold_mv * 1.0e-3
        if self.floating_k is None:
            return np.full(x.shape[-1], fixed)
        # causal trailing RMS, lagged by one window so the event itself does not raise it
        e = np.concatenate([[0.0], np.cumsum(x**2)])
        n = self.noise_window
        idx = np.arange(x.shape[-1])
        lo = np.maximum(idx - 2 * n, 0)
        hi = np.maximum(idx - n, 1)
        rms = np.sqrt((e[hi] - e[lo]) / np.maximum(hi - lo, 1))
        return np.maximum(fixed, self.floating_k * rms)

    # ------------------------------------------------------------------ single trigger
    def evaluate_trigger(self, calibrated_volts: np.ndarray) -> Tuple[bool, int, int]:
        """First threshold crossing over all channels: (triggered, channel, sample_idx)."""
        v = np.asarray(calibrated_volts, dtype=np.float64)
        if v.ndim == 1:
            v = v[None, :]
        earliest_idx, earliest_ch = v.shape[1] + 1, -1
        for ch in range(v.shape[0]):
            crossings = np.where(np.abs(v[ch]) >= self.threshold_volts(v[ch]))[0]
            if len(crossings) > 0 and crossings[0] < earliest_idx:
                earliest_idx, earliest_ch = int(crossings[0]), ch
        if earliest_ch >= 0:
            return True, earliest_ch, earliest_idx
        return False, 0, -1

    # ------------------------------------------------------------------ multi-hit
    def find_hits(self, calibrated_volts: np.ndarray) -> List[Tuple[int, int]]:
        """All hits in a record as (sample_idx, first_channel), separated by the dead time."""
        v = np.asarray(calibrated_volts, dtype=np.float64)
        if v.ndim == 1:
            v = v[None, :]
        above = np.zeros(v.shape[1], dtype=bool)
        first_ch = np.full(v.shape[1], -1)
        for ch in range(v.shape[0]):
            m = np.abs(v[ch]) >= self.threshold_volts(v[ch])
            first_ch = np.where(m & (first_ch < 0), ch, first_ch)
            above |= m
        hits = []
        i = 0
        idx = np.where(above)[0]
        while i < len(idx):
            s = int(idx[i])
            hits.append((s, int(first_ch[s])))
            nxt = s + max(self.dead_samples, 1)
            i = int(np.searchsorted(idx, nxt))
        return hits

    def extract_event_window(self, waveforms: np.ndarray, trigger_sample_idx: int) -> np.ndarray:
        """Pre + post trigger window, zero-padded at the record boundaries."""
        n_channels, total_len = waveforms.shape
        pre, post = self.config.pre_samples, self.config.post_samples
        start, end = trigger_sample_idx - pre, trigger_sample_idx + post
        window = np.zeros((n_channels, pre + post), dtype=waveforms.dtype)
        src_start, src_end = max(0, start), min(total_len, end)
        dst_start = max(0, -start)
        window[:, dst_start : dst_start + (src_end - src_start)] = waveforms[:, src_start:src_end]
        return window
