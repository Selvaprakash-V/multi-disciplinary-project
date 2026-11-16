"""Out-of-distribution ("unknown source") detection.

Field noise will include sources the simulator never produced.  Forcing every event into a
known class hides that; instead events far from all training classes are reported as
"unknown".  Class-conditional Gaussian / Mahalanobis scoring with a shared covariance
(Lee et al. 2018) on standardised features.
"""

from typing import Optional

import numpy as np


class MahalanobisOOD:
    def __init__(self, quantile: float = 0.99, shrinkage: float = 1e-2):
        self.quantile = quantile
        self.shrinkage = shrinkage
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.prec: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    def _z(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mu) / self.sd

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MahalanobisOOD":
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Z = self._z(X)
        classes = np.unique(y)
        self.means = np.stack([Z[y == c].mean(axis=0) for c in classes])
        centered = np.concatenate([Z[y == c] - self.means[i] for i, c in enumerate(classes)])
        cov = np.cov(centered, rowvar=False)
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.eye(cov.shape[0])
        self.prec = np.linalg.pinv(cov)
        self.threshold = float(np.quantile(self.score(X), self.quantile))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Minimum Mahalanobis distance to any class mean (higher = more unusual)."""
        Z = self._z(X)
        d = []
        for m in self.means:
            diff = Z - m
            d.append(np.einsum("ij,jk,ik->i", diff, self.prec, diff))
        return np.sqrt(np.min(np.stack(d, axis=1), axis=1))

    def is_unknown(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold
