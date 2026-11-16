from typing import Optional, Tuple

import numpy as np


class OnsetPicker:
    """
    Waveform onset (arrival-time) pickers:
    1. AIC (Maeda 1985) — vectorised O(N) with cumulative sums.
    2. STA/LTA — causal trailing windows (realisable in real time on the edge node).
    3. ``threshold_onset`` — first crossing of k x robust noise level.
    """

    @staticmethod
    def pick_aic(signal: np.ndarray, search_window: Optional[Tuple[int, int]] = None) -> int:
        """Maeda AIC onset pick:
            AIC(k) = k log(var(x[w1:k])) + (N-k) log(var(x[k:w2]))
        computed for all k at once from cumulative sums."""
        x = np.asarray(signal, dtype=np.float64)
        N = len(x)
        if N < 20:
            return 0
        w1 = 0 if search_window is None else max(0, search_window[0])
        w2 = N if search_window is None else min(N, search_window[1])
        margin = 5
        if (w2 - w1) <= 2 * margin:
            return w1
        seg = x[w1:w2]
        n = len(seg)
        c1 = np.cumsum(seg)
        c2 = np.cumsum(seg**2)
        k = np.arange(margin, n - margin)
        m1 = c1[k - 1] / k
        v1 = c2[k - 1] / k - m1**2
        m2 = (c1[-1] - c1[k - 1]) / (n - k)
        v2 = (c2[-1] - c2[k - 1]) / (n - k) - m2**2
        eps = 1e-30
        aic = k * np.log(np.maximum(v1, eps)) + (n - k) * np.log(np.maximum(v2, eps))
        return int(w1 + k[np.argmin(aic)])

    @staticmethod
    def pick_sta_lta(signal: np.ndarray, n_sta: int = 20, n_lta: int = 150, threshold: float = 3.5) -> int:
        """Causal STA/LTA: trailing short window over trailing long window (which ends where
        the short window starts).  Returns the first sample where the ratio exceeds
        ``threshold`` (or the ratio maximum if it never does)."""
        x = np.asarray(signal, dtype=np.float64)
        N = len(x)
        if N <= n_sta + n_lta:
            return 0
        e = np.concatenate([[0.0], np.cumsum(x**2)])
        idx = np.arange(n_sta + n_lta, N + 1)
        sta = (e[idx] - e[idx - n_sta]) / n_sta
        lta = (e[idx - n_sta] - e[idx - n_sta - n_lta]) / n_lta
        ratio = sta / (lta + 1e-30)
        hits = np.where(ratio >= threshold)[0]
        j = int(hits[0]) if len(hits) else int(np.argmax(ratio))
        # ratio at index j covers samples ending at idx[j]-1; the onset lies inside the STA window
        return int(idx[j] - n_sta // 2)

    @staticmethod
    def robust_noise_sigma(signal: np.ndarray, n_segments: int = 16) -> float:
        """Noise sigma estimated from the quietest segments (robust to where the event is)."""
        x = np.asarray(signal, dtype=np.float64)
        segs = np.array_split(x, n_segments)
        rms = np.sort([np.sqrt(np.mean(s**2)) for s in segs if len(s)])
        return float(np.mean(rms[: max(1, len(rms) // 4)]))

    @staticmethod
    def threshold_onset(signal: np.ndarray, k_sigma: float = 6.0, sigma: Optional[float] = None) -> int:
        """First sample whose |x| exceeds k_sigma x noise sigma (-1 if none)."""
        x = np.asarray(signal, dtype=np.float64)
        s = OnsetPicker.robust_noise_sigma(x) if sigma is None else sigma
        hits = np.where(np.abs(x) > k_sigma * max(s, 1e-30))[0]
        return int(hits[0]) if len(hits) else -1
