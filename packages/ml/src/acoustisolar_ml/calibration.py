"""Probability calibration: temperature scaling and split-conformal prediction sets."""

from typing import List, Optional

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaler:
    """Single-parameter temperature scaling of multi-class probabilities (Guo et al. 2017).

    Fitted on a held-out calibration split by minimising negative log-likelihood.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        logits = np.log(np.clip(proba, 1e-12, 1.0))
        y = np.asarray(y, dtype=int)
        best_t, best_nll = 1.0, np.inf
        for t in np.exp(np.linspace(np.log(0.05), np.log(20.0), 200)):
            p = _softmax(logits / t)
            nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))
            if nll < best_nll:
                best_t, best_nll = t, nll
        self.temperature = float(best_t)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return _softmax(np.log(np.clip(proba, 1e-12, 1.0)) / self.temperature)


class ConformalClassifier:
    """Split-conformal prediction sets (LAC score 1 - p_true).

    Guarantees marginal coverage >= 1 - alpha on exchangeable data, so an alert can state
    "the true source is in {crack_growth, hail_crack}" with a known error rate.  Coverage
    holds for the simulated distribution only until re-calibrated on real data.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat: Optional[float] = None

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "ConformalClassifier":
        y = np.asarray(y, dtype=int)
        scores = 1.0 - proba[np.arange(len(y)), y]
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, proba: np.ndarray) -> List[List[int]]:
        if self.qhat is None:
            raise RuntimeError("ConformalClassifier.fit must be called first")
        return [list(np.where(1.0 - p <= self.qhat)[0]) for p in proba]

    @staticmethod
    def coverage(sets: List[List[int]], y: np.ndarray) -> float:
        return float(np.mean([int(t) in s for s, t in zip(sets, y)]))
