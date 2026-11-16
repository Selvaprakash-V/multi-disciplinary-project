from acoustisolar import AcoustiSolar


def test_sdk_simulate_event():
    client = AcoustiSolar(seed=42)
    res = client.simulate_event(x_mm=500.0, y_mm=800.0, energy=3.0)
    assert res["triggered"] is True
    assert res["calibrated_volts"].shape[0] == 4
    loc = res["localization"]
    assert loc is not None
    assert abs(loc["x_mm"] - 500.0) < 50.0 and abs(loc["y_mm"] - 800.0) < 50.0


def test_sdk_generate_dataset():
    client = AcoustiSolar(seed=42)
    waves, df = client.generate_synthetic_dataset(n_events=4)
    assert waves.shape[0] == 4
    assert len(df) == 4
    assert "event_class" in df.columns
