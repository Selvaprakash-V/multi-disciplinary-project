"""Feature-based classifiers.

* ``AEBaselineGradientBooster`` — tiny dependency-free binary booster (depth-1 stumps);
  kept as an always-available edge fallback and reference.
* ``AEFeatureClassifier`` — multi-class gradient-boosted trees (scikit-learn
  ``HistGradientBoostingClassifier``) on the engineered AE feature vector, with
  hash-verified persistence.
"""

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from acoustisolar_schemas.event import EVENT_CLASSES, NEW_FRACTURE_CLASSES


class DecisionStump:
    """A single threshold decision stump."""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_val = 0.0
        self.right_val = 0.0

    def fit(self, X: np.ndarray, residuals: np.ndarray):
        best_loss = float("inf")
        for feat in range(X.shape[1]):
            vals = X[:, feat]
            for thresh in np.percentile(vals, np.linspace(10, 90, 9)):
                left = vals <= thresh
                if not np.any(left) or np.all(left):
                    continue
                lm, rm = float(np.mean(residuals[left])), float(np.mean(residuals[~left]))
                loss = float(np.sum((residuals - np.where(left, lm, rm)) ** 2))
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx, self.threshold = feat, float(thresh)
                    self.left_val, self.right_val = lm, rm

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, self.feature_idx] <= self.threshold, self.left_val, self.right_val)


class AEBaselineGradientBooster:
    """Binary gradient-boosted stumps (log-loss).  Dependency-free reference model."""

    def __init__(self, n_estimators: int = 25, learning_rate: float = 0.15):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.trees = []
        self.base_pred = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_pred = np.log(p / (1.0 - p))
        f = np.full(len(y), self.base_pred)
        self.trees = []
        for _ in range(self.n_estimators):
            residuals = y - 1.0 / (1.0 + np.exp(-f))
            tree = DecisionStump()
            tree.fit(X, residuals)
            f += self.lr * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        f = np.full(len(X), self.base_pred)
        for tree in self.trees:
            f += self.lr * tree.predict(X)
        return 1.0 / (1.0 + np.exp(-f))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class AEFeatureClassifier:
    """Multi-class source classifier on engineered features (gradient-boosted trees)."""

    def __init__(self, classes: Sequence[str] = EVENT_CLASSES, max_iter: int = 300, learning_rate: float = 0.08,
                 random_state: int = 42):
        self.classes = list(classes)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        from sklearn.ensemble import HistGradientBoostingClassifier

        y = np.asarray(y, int)
        # Early stopping needs a stratified validation split: only with enough data per class
        counts = np.bincount(y)
        use_es = len(y) >= 500 and counts[counts > 0].min() >= 10
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=self.learning_rate, random_state=self.random_state,
            class_weight="balanced", early_stopping=use_es, validation_fraction=0.15 if use_es else None,
        )
        self.model.fit(np.asarray(X, float), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilities over ``self.classes`` (columns for classes unseen in training are 0)."""
        raw = self.model.predict_proba(np.asarray(X, float))
        out = np.zeros((raw.shape[0], len(self.classes)))
        out[:, self.model.classes_] = raw
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def new_fracture_probability(self, proba: np.ndarray) -> np.ndarray:
        idx = [i for i, c in enumerate(self.classes) if c in NEW_FRACTURE_CLASSES]
        return proba[:, idx].sum(axis=1)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> str:
        """Save with joblib and write a SHA-256 sidecar; returns the digest."""
        import joblib

        path = Path(path)
        joblib.dump({"classes": self.classes, "model": self.model}, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
        return digest

    @classmethod
    def load(cls, path, expected_sha256: Optional[str] = None) -> "AEFeatureClassifier":
        """Load only if the file's SHA-256 matches the sidecar (or ``expected_sha256``).

        joblib/pickle files can execute code when loaded; never load a model file whose
        provenance you have not verified.
        """
        import joblib

        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is None:
            sidecar = path.with_suffix(path.suffix + ".sha256")
            if not sidecar.exists():
                raise ValueError(f"Refusing to load {path}: no SHA-256 sidecar to verify it against")
            expected_sha256 = sidecar.read_text(encoding="utf-8").strip()
        if digest != expected_sha256:
            raise ValueError(f"Refusing to load {path}: SHA-256 mismatch")
        payload = joblib.load(path)
        obj = cls(classes=payload["classes"])
        obj.model = payload["model"]
        return obj
