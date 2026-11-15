import numpy as np
from typing import Dict, Any, Tuple

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute binary classification metrics: Precision, Recall, F1, and ECE.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-6, (precision + recall))
    accuracy = (tp + tn) / max(1, len(y_true))
    
    # Expected Calibration Error (ECE) with 10 bins
    n_bins = 10
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = (y_prob >= bin_limits[b]) & (y_prob < bin_limits[b + 1])
        if np.any(in_bin):
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            bin_weight = np.sum(in_bin) / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)
            
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "ece": float(ece),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

def compute_localization_metrics(
    true_xy_mm: np.ndarray,
    pred_xy_mm: np.ndarray,
    true_cell_ids: list,
    pred_cell_ids: list,
) -> Dict[str, float]:
    """
    Compute localization metrics:
    - Median error (mm)
    - 90th percentile error (mm)
    - Cell ID classification accuracy
    """
    errors_mm = np.sqrt(np.sum((true_xy_mm - pred_xy_mm)**2, axis=1))
    median_err = float(np.median(errors_mm))
    pct90_err = float(np.percentile(errors_mm, 90))
    
    matches = sum(1 for t, p in zip(true_cell_ids, pred_cell_ids) if t == p)
    cell_acc = float(matches / max(1, len(true_cell_ids)))
    
    return {
        "median_error_mm": median_err,
        "p90_error_mm": pct90_err,
        "cell_id_accuracy": cell_acc,
    }

