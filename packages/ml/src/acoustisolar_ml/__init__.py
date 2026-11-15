"""AcoustiSolar ML: Dataset generation, baseline tree models, 1D ResNet, metrics, and held-out evaluation."""

from .generator import DatasetGenerator
from .baseline_model import AEBaselineGradientBooster
from .deep_model import Waveform1DResNet
from .metrics import compute_classification_metrics, compute_localization_metrics
from .evaluation import HeldOutEvaluator

__all__ = [
    "DatasetGenerator",
    "AEBaselineGradientBooster",
    "Waveform1DResNet",
    "compute_classification_metrics",
    "compute_localization_metrics",
    "HeldOutEvaluator",
]

