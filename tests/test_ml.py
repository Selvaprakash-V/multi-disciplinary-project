import numpy as np
import pytest

from acoustisolar_ml import (
    AEBaselineGradientBooster,
    ConformalClassifier,
    DatasetGenerator,
    HeldOutEvaluator,
    MahalanobisOOD,
    TemperatureScaler,
    compute_classification_metrics,
    evt_threshold,
    expected_calibration_error,
    feature_matrix,
    label_vector,
    multiclass_metrics,
    pr_auc,
)
from acoustisolar_schemas.event import EVENT_CLASSES


@pytest.fixture(scope="module")
def small_batch():
    return DatasetGenerator(seed=123).generate_batch(n_events=24, crack_ratio=0.5)


def test_dataset_generator_labels_and_shapes(small_batch):
    waves, df = small_batch
    assert waves.shape == (24, 4, 2560)
    assert set(df["event_class"]).issubset(EVENT_CLASSES)
    assert {"is_crack", "is_crack_activity", "loc_error_mm", "cell_id", "loc_cell_id"} <= set(df.columns)
    # friction is crack *activity* but not a *new* fracture
    fr = df[df["event_class"] == "crack_friction"]
    assert (fr["is_crack"] == 0).all() and (fr["is_crack_activity"] == 1).all()
    assert len(df["config_hash"].iloc[0]) == 64


def test_records_have_noise_only_pretrigger(small_batch):
    waves, df = small_batch
    # events start >= 300 us after record start: the first 200 samples are noise only
    for w, onset in zip(waves, df["onset_us"]):
        assert onset >= 300.0
    pre = np.abs(waves[:, :, :200]).max()
    assert pre < 0.5 * np.abs(waves).max()


def test_frame_sources_are_located_at_edges():
    g = DatasetGenerator(seed=5, localize=False)
    rng = np.random.default_rng(0)
    W, L = g.module_config.width_mm, g.module_config.length_mm
    for _ in range(20):
        x, y = g._random_location("thermal_tick", rng)
        assert min(x, W - x, y, L - y) <= 25.0


def test_feature_vector_length_matches_names(small_batch):
    from acoustisolar_dsp.features import AEFeatureExtractor

    _, df = small_batch
    assert len(df["feature_vector"].iloc[0]) == len(AEFeatureExtractor.feature_names(4))


def test_baseline_gradient_booster():
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(-1.5, 0.5, (30, 8)), rng.normal(1.5, 0.5, (30, 8))])
    y = np.array([0] * 30 + [1] * 30)
    model = AEBaselineGradientBooster(n_estimators=15, learning_rate=0.2).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.85


def test_feature_classifier_multiclass_and_safe_persistence(tmp_path, small_batch):
    pytest.importorskip("sklearn")
    from acoustisolar_ml import AEFeatureClassifier

    _, df = small_batch
    X, y = feature_matrix(df), label_vector(df)
    clf = AEFeatureClassifier(max_iter=30).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), len(EVENT_CLASSES))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    path = tmp_path / "clf.joblib"
    clf.save(path)
    loaded = AEFeatureClassifier.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), proba)
    # tampered file must be refused
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        AEFeatureClassifier.load(path)


def test_waveform_resnet_trains_and_has_residuals(small_batch):
    pytest.importorskip("torch")
    from acoustisolar_ml import Waveform1DResNet

    waves, df = small_batch
    y = label_vector(df)
    net = Waveform1DResNet(width=8, device="cpu")
    before = [p.detach().clone() for p in net.net.parameters()]
    net.fit(waves, y, epochs=2, batch_size=8)
    after = list(net.net.parameters())
    assert any(not np.allclose(b.numpy(), a.detach().numpy()) for b, a in zip(before, after))  # it learns
    proba = net.predict_proba(waves[:3])
    assert proba.shape == (3, len(EVENT_CLASSES))
    assert any(type(m).__name__ == "Block" for m in net.net.modules())


def test_metrics_ece_counts_probability_one():
    assert expected_calibration_error(np.array([1, 1]), np.array([1.0, 1.0])) == 0.0
    assert expected_calibration_error(np.array([0, 0]), np.array([1.0, 1.0])) == pytest.approx(1.0)


def test_pr_auc_and_binary_metrics():
    y = np.array([0, 0, 1, 1])
    assert pr_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    m = compute_classification_metrics(y, np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["f1_score"] == pytest.approx(1.0)


def test_multiclass_metrics_confusion():
    m = multiclass_metrics(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]), ["a", "b", "c"])
    assert m["confusion_matrix"][2][1] == 1
    assert 0.0 < m["macro_f1"] < 1.0


def test_evt_threshold_extrapolates_tail():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    scores = rng.exponential(1.0, 5000)
    out = evt_threshold(scores, 1e-4)
    assert out["method"] == "gpd"
    # exponential tail: true 1e-4 quantile = ln(1e4) ~ 9.2
    assert 7.0 < out["threshold"] < 12.0


def test_temperature_scaling_and_conformal_coverage():
    rng = np.random.default_rng(1)
    n, k = 2000, 4
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.5
    over = np.exp(4 * logits)
    proba = over / over.sum(axis=1, keepdims=True)  # over-confident
    ts = TemperatureScaler().fit(proba[:1000], y[:1000])
    assert ts.temperature > 1.0
    cal = ts.transform(proba)
    conf = ConformalClassifier(alpha=0.1).fit(cal[:1000], y[:1000])
    sets = conf.predict_sets(cal[1000:])
    assert ConformalClassifier.coverage(sets, y[1000:]) >= 0.87


def test_mahalanobis_flags_unseen_cluster():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0, 1, (200, 5)), rng.normal(5, 1, (200, 5))])
    y = np.array([0] * 200 + [1] * 200)
    ood = MahalanobisOOD().fit(X, y)
    novel = rng.normal(-12, 1, (50, 5))
    assert ood.is_unknown(novel).mean() > 0.9
    assert ood.is_unknown(X).mean() < 0.05


def test_held_out_evaluation():
    metrics = HeldOutEvaluator(seed=99).run_held_out_benchmark(n_train=30, n_test=15)
    assert {"accuracy", "f1_score", "ece", "pr_auc"} <= set(metrics)
    assert metrics["train_module"] != metrics["held_out_module"]
