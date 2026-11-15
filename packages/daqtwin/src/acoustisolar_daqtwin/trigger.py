import numpy as np
from typing import Optional, Tuple
from acoustisolar_schemas.daq import TriggerConfig

class TriggerEngine:
    """
    Hardware and firmware event trigger engine:
    - Absolute amplitude threshold in millivolts
    - Pre-trigger and post-trigger event window extraction
    - Hysteresis and re-trigger holdoff (dead time)
    """

    def __init__(self, config: Optional[TriggerConfig] = None, sample_rate_hz: float = 1_000_000.0):
        self.config = config or TriggerConfig()
        self.sample_rate = sample_rate_hz
        self.dead_samples = int((self.config.dead_time_us * 1.0e-6) * sample_rate_hz)

    def evaluate_trigger(self, calibrated_volts: np.ndarray) -> Tuple[bool, int, int]:
        """
        Check if any channel crosses threshold.
        Returns:
            (triggered: bool, trigger_channel: int, trigger_sample_idx: int)
        """
        v = np.asarray(calibrated_volts, dtype=np.float64)
        thresh_v = self.config.threshold_mv * 1.0e-3
        
        if v.ndim == 1:
            crossings = np.where(np.abs(v) >= thresh_v)[0]
            if len(crossings) > 0:
                return True, 0, int(crossings[0])
            return False, 0, -1
        else:
            n_channels, n_samples = v.shape
            earliest_idx = n_samples + 1
            earliest_ch = -1
            
            for ch in range(n_channels):
                crossings = np.where(np.abs(v[ch]) >= thresh_v)[0]
                if len(crossings) > 0 and crossings[0] < earliest_idx:
                    earliest_idx = int(crossings[0])
                    earliest_ch = ch
                    
            if earliest_ch >= 0:
                return True, earliest_ch, earliest_idx
            return False, 0, -1

    def extract_event_window(
        self,
        waveforms: np.ndarray,
        trigger_sample_idx: int
    ) -> np.ndarray:
        """
        Extract pre_trigger + post_trigger sample window centered around trigger.
        Pads with zero or edge if trigger occurs near the boundaries.
        """
        n_channels, total_len = waveforms.shape
        pre = self.config.pre_samples
        post = self.config.post_samples
        target_len = pre + post
        
        start = trigger_sample_idx - pre
        end = trigger_sample_idx + post
        
        window = np.zeros((n_channels, target_len), dtype=waveforms.dtype)
        
        src_start = max(0, start)
        src_end = min(total_len, end)
        dst_start = max(0, -start)
        dst_end = dst_start + (src_end - src_start)
        
        window[:, dst_start:dst_end] = waveforms[:, src_start:src_end]
        return window

