import pytest
from acoustisolar import AcoustiSolar
from acoustisolar_schemas.source import SourceClass, CrackType

def test_sdk_simulate_event():
    client = AcoustiSolar(seed=42)
    res = client.simulate_event(x_mm=500.0, y_mm=800.0)
    assert res["triggered"] is True
    assert res["calibrated_volts"].shape[0] == 4
    if res["localization"]:
        assert abs(res["localization"]["x_mm"] - 500.0) < 50.0

def test_sdk_generate_dataset():
    client = AcoustiSolar(seed=42)
    waves, df = client.generate_synthetic_dataset(n_events=4)
    assert waves.shape[0] == 4
    assert len(df) == 4

