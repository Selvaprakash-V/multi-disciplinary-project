"""Evaluation metrics: classification, calibration, false-alarm rate and localisation."""

from typing import Dict, List, Sequence

import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Binary ECE with equal-width bins; the last bin is closed so p = 1.0 is counted."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        hi_ok = y_prob <= edges[b + 1] if b == n_bins - 1 else y_prob < edges[b + 1]
        in_bin = (y_prob >= edges[b]) & hi_ok
        if np.any(in_bin):
            ece += np.mean(in_bin) * abs(np.mean(y_true[in_bin]) - np.mean(y_prob[in_bin]))
    return float(ece)


def multiclass_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE for multi-class probabilities."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(y_true)).astype(float)
    return expected_calibration_error(correct, conf, n_bins)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Binary metrics: precision, recall, F1, accuracy, ECE, PR-AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "ece": expected_calibration_error(y_true, y_prob),
        "pr_auc": pr_auc(y_true, y_prob),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(y_score, dtype=float))
    y = y_true[order]
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision[y == 1]) / n_pos)


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str]) -> Dict[str, object]:
    """Macro-F1, per-class F1 and confusion matrix (rows = true, cols = predicted)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < k and 0 <= p < k:
            cm[t, p] += 1
    per_class = {}
    f1s = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        if cm[i, :].sum() == 0:
            continue  # class absent from the test set: excluded from the macro average
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        per_class[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(cm[i].sum())}
        f1s.append(f1)
    return {
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
    }


def far_curve(noise_scores: np.ndarray, thresholds: np.ndarray, noise_triggers_per_day: float) -> np.ndarray:
    """False alarms per module-day vs decision threshold (empirical)."""
    s = np.asarray(noise_scores, dtype=float)
    return np.array([np.mean(s >= t) * noise_triggers_per_day for t in thresholds])


def evt_threshold(noise_scores: np.ndarray, target_fpr: float, tail_fraction: float = 0.1) -> Dict[str, float]:
    """Decision threshold for a very small false-positive rate via a Generalised Pareto fit
    to the upper tail of the noise-score distribution (peaks-over-threshold).

    The empirical 99.99th percentile of a few thousand noise events is effectively the
    sample maximum and is not a meaningful estimate; the GPD tail model extrapolates in a
    principled way (still only as good as the simulated noise model).
    """
    from scipy.stats import genpareto

    s = np.sort(np.asarray(noise_scores, dtype=float))
    n = len(s)
    u = float(np.quantile(s, 1.0 - tail_fraction))
    exc = s[s > u] - u
    p_u = len(exc) / n
    if len(exc) < 20 or target_fpr >= p_u:
        return {"threshold": float(np.quantile(s, 1.0 - target_fpr)), "method": "empirical",
                "n_tail": int(len(exc))}
    c, _loc, scale = genpareto.fit(exc, floc=0.0)
    q = genpareto.ppf(1.0 - target_fpr / p_u, c, loc=0.0, scale=scale)
    return {"threshold": float(u + q), "method": "gpd", "shape": float(c), "scale": float(scale),
            "n_tail": int(len(exc))}


def compute_localization_metrics(
    true_xy_mm: np.ndarray,
    pred_xy_mm: np.ndarray,
    true_cell_ids: List[str],
    pred_cell_ids: List[str],
) -> Dict[str, float]:
    """Median / 90th-percentile error (mm) and cell-ID accuracy (ids from the module layout)."""
    errors = np.sqrt(np.sum((np.asarray(true_xy_mm) - np.asarray(pred_xy_mm)) ** 2, axis=1))
    ok = np.isfinite(errors)
    matches = sum(1 for t, p in zip(true_cell_ids, pred_cell_ids) if t == p)
    return {
        "median_error_mm": float(np.median(errors[ok])) if ok.any() else float("nan"),
        "p90_error_mm": float(np.percentile(errors[ok], 90)) if ok.any() else float("nan"),
        "cell_id_accuracy": float(matches / max(1, len(true_cell_ids))),
        "n_localized": int(ok.sum()),
    }


def bootstrap_ci(metric_fn, *arrays, n_boot: int = 500, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap CI of ``metric_fn(*resampled_arrays)``."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(*[np.asarray(a)[idx] for a in arrays]))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
