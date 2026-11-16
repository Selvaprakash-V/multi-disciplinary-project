"""AcoustiSolar ML: dataset generation, feature and waveform classifiers, calibration,
unknown-source detection, metrics and held-out evaluation."""

from .baseline_model import AEBaselineGradientBooster, AEFeatureClassifier
from .calibration import ConformalClassifier, TemperatureScaler
from .evaluation import HeldOutEvaluator
from .generator import DatasetGenerator, class_probabilities, feature_matrix, label_vector
from .metrics import (
    compute_classification_metrics,
    compute_localization_metrics,
    evt_threshold,
    expected_calibration_error,
    far_curve,
    multiclass_metrics,
    pr_auc,
)
from .ood import MahalanobisOOD


def __getattr__(name):
    # PyTorch is optional: import the deep model lazily
    if name == "Waveform1DResNet":
        from .deep_model import Waveform1DResNet

        return Waveform1DResNet
    raise AttributeError(name)


__all__ = [
    "DatasetGenerator",
    "class_probabilities",
    "feature_matrix",
    "label_vector",
    "AEBaselineGradientBooster",
    "AEFeatureClassifier",
    "Waveform1DResNet",
    "TemperatureScaler",
    "ConformalClassifier",
    "MahalanobisOOD",
    "compute_classification_metrics",
    "compute_localization_metrics",
    "multiclass_metrics",
    "expected_calibration_error",
    "pr_auc",
    "far_curve",
    "evt_threshold",
    "HeldOutEvaluator",
]
