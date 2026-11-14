import numpy as np
from typing import Dict, Any, Tuple, Optional

class DecisionStump:
    """A single threshold decision stump for gradient boosting / decision trees."""
    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_val = 0.0
        self.right_val = 0.0
        self.polarity = 1

    def fit(self, X: np.ndarray, residuals: np.ndarray):
        n_samples, n_features = X.shape
        best_loss = float("inf")
        
        # Test candidate percentiles for speed and robustness
        for feat in range(n_features):
            vals = X[:, feat]
            thresholds = np.percentile(vals, np.linspace(10, 90, 9))
            for thresh in thresholds:
                left_mask = vals <= thresh
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                left_mean = float(np.mean(residuals[left_mask]))
                right_mean = float(np.mean(residuals[right_mask]))
                
                pred = np.where(left_mask, left_mean, right_mean)
                loss = float(np.sum((residuals - pred)**2))
                
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = feat
                    self.threshold = float(thresh)
                    self.left_val = left_mean
                    self.right_val = right_mean

    def predict(self, X: np.ndarray) -> np.ndarray:
        mask = X[:, self.feature_idx] <= self.threshold
        return np.where(mask, self.left_val, self.right_val)

class AEBaselineGradientBooster:
    """
    Gradient Boosted Trees baseline classifier for AE features.
    Classifies crack vs. noise and outputs calibrated probability.
    """

    def __init__(self, n_estimators: int = 25, learning_rate: float = 0.15):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.trees = []
        self.base_pred = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit gradient booster on feature matrix X and binary labels y in {0, 1}."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        p = np.mean(y)
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        self.base_pred = np.log(p / (1.0 - p))
        
        f = np.full(len(y), self.base_pred)
        self.trees = []
        
        for _ in range(self.n_estimators):
            # Compute negative gradient of log-loss: y - p
            prob = 1.0 / (1.0 + np.exp(-f))
            residuals = y - prob
            
            tree = DecisionStump()
            tree.fit(X, residuals)
            step = tree.predict(X)
            f += self.lr * step
            self.trees.append(tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute calibrated probability P(y=1) for test samples."""
        X = np.asarray(X, dtype=np.float64)
        f = np.full(len(X), self.base_pred)
        for tree in self.trees:
            f += self.lr * tree.predict(X)
        probs = 1.0 / (1.0 + np.exp(-f))
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
