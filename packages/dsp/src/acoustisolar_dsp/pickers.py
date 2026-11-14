import numpy as np
from typing import Optional, Tuple

class OnsetPicker:
    """
    Waveform onset arrival-time pickers:
    1. AIC (Akaike Information Criterion): mathematically rigorous auto-picker.
    2. STA/LTA (Short-Term Average / Long-Term Average): classic seismic & AE trigger.
    """

    @staticmethod
    def pick_aic(signal: np.ndarray, search_window: Optional[Tuple[int, int]] = None) -> int:
        """
        Compute Maeda (1985) Akaike Information Criterion (AIC) onset pick on 1D waveform.
        Evaluates variance change within search window [w1, w2]:
            AIC(k) = (k - w1) * log(var(x[w1..k])) + (w2 - k) * log(var(x[k..w2]))
        The global minimum indicates the onset time.
        """
        x = np.asarray(signal, dtype=np.float64)
        N = len(x)
        if N < 20:
            return 0
            
        w1 = 0 if search_window is None else max(0, search_window[0])
        w2 = N if search_window is None else min(N, search_window[1])
        
        margin = 5
        if (w2 - w1) <= 2 * margin:
            return w1

        aic = []
        eps = 1e-15
        eval_range = range(w1 + margin, w2 - margin)
        
        for k in eval_range:
            seg1 = x[w1:k]
            seg2 = x[k:w2]
            var1 = np.var(seg1) + eps
            var2 = np.var(seg2) + eps
            val = (k - w1) * np.log(var1) + (w2 - k) * np.log(var2)
            aic.append(val)
            
        pick = int(eval_range[np.argmin(aic)])
        return pick

    @staticmethod
    def pick_sta_lta(
        signal: np.ndarray,
        n_sta: int = 20,
        n_lta: int = 150,
        threshold: float = 3.5
    ) -> int:
        """
        Compute STA/LTA energy ratio and return the first sample crossing threshold.
        """
        x = np.asarray(signal, dtype=np.float64)
        energy = x**2
        N = len(x)
        
        if N <= n_lta:
            return 0
            
        # Recursive / moving window STA and LTA
        sta = np.convolve(energy, np.ones(n_sta) / n_sta, mode='same')
        lta = np.convolve(energy, np.ones(n_lta) / n_lta, mode='same')
        
        eps = 1e-12
        ratio = sta / (lta + eps)
        
        crossings = np.where(ratio[n_lta:] >= threshold)[0]
        if len(crossings) > 0:
            return int(crossings[0] + n_lta)
        return int(np.argmax(ratio))
