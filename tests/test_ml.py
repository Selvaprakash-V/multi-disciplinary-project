import pytest
import numpy as np
from acoustisolar_ml import (
    DatasetGenerator,
    AEBaselineGradientBooster,
    Waveform1DResNet,
    compute_classification_metrics,
    HeldOutEvaluator,
)
from acoustisolar_schemas.source import SourceClass, CrackType

def test_dataset_generator():
    gen = DatasetGenerator(seed=123)
    waves, df = gen.generate_batch(n_events=6, crack_ratio=0.5)
    assert waves.shape == (6, 4, 2560)
    assert len(df) == 6
    assert "is_crack" in df.columns
    assert "config_hash" in df.columns
    assert len(df["config_hash"].iloc[0]) == 64

def test_baseline_gradient_booster():
    # Synthetic feature dataset
    rng = np.random.default_rng(42)
    n = 60
    # Two distinct clusters
    X0 = rng.normal(loc=-1.5, scale=0.5, size=(n // 2, 8))
    X1 = rng.normal(loc=1.5, scale=0.5, size=(n // 2, 8))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    
    model = AEBaselineGradientBooster(n_estimators=15, learning_rate=0.2)
    model.fit(X, y)
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    assert len(probs) == n
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    acc = np.mean(preds == y)
    assert acc > 0.85

def test_waveform_1d_resnet():
    model = Waveform1DResNet(in_channels=4, n_features=8, seed=42)
    batch = np.random.normal(0, 1.0, (3, 4, 1000)).astype(np.float32)
    probs = model.forward(batch)
    assert probs.shape == (3,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    preds = model.predict(batch)
    assert len(preds) == 3

def test_held_out_evaluation():
    evaluator = HeldOutEvaluator(seed=99)
    metrics = evaluator.run_held_out_benchmark(n_train=20, n_test=10)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "ece" in metrics
    assert metrics["train_module"] != metrics["held_out_module"]

