import pytest
import tempfile
from pathlib import Path
from acoustisolar_schemas import (
    ModuleTwinConfig,
    LayerConfig,
    SensorArrayConfig,
    DAQConfig,
    SourceEventConfig,
    SourceClass,
    CrackType,
    EventRecord,
    TriggerFeatures,
    GroundTruthLabel,
    ProvenanceRecord,
    compute_config_hash,
    export_json_schemas,
)

def test_module_twin_config():
    config = ModuleTwinConfig()
    assert config.layout.cells == 120
    assert len(config.stack) == 5
    assert pytest.approx(config.total_thickness_mm(), 0.01) == 4.575

def test_config_hashing_determinism():
    c1 = ModuleTwinConfig()
    c2 = ModuleTwinConfig()
    h1 = compute_config_hash(c1)
    h2 = compute_config_hash(c2)
    assert h1 == h2
    assert len(h1) == 64

    # Changing a parameter must alter the hash
    c2.stack[0].thickness_mm = 3.5
    h3 = compute_config_hash(c2)
    assert h1 != h3

def test_event_record_validation():
    event = EventRecord(
        trigger_features=TriggerFeatures(
            peak_amp_mv=125.4,
            rise_time_us=8.5,
            duration_us=180.0,
            counts=34,
            energy_au=4500.0,
            centroid_khz=165.2,
        ),
        label=GroundTruthLabel(
            source_class=SourceClass.CRACK_INITIATION,
            crack_type=CrackType.DIAGONAL,
            x_mm=300.0,
            y_mm=750.0,
        ),
        provenance=ProvenanceRecord(
            sim_tier="T0",
            config_hash="abc12345",
            seed=42,
        ),
    )
    assert event.trigger_features.peak_amp_mv == 125.4
    assert event.label.crack_type == CrackType.DIAGONAL
    assert event.n_channels == 4

def test_export_json_schemas():
    with tempfile.TemporaryDirectory() as tmpdir:
        exported = export_json_schemas(Path(tmpdir))
        assert "ModuleTwinConfig" in exported
        assert "EventRecord" in exported
        assert (Path(tmpdir) / "ModuleTwinConfig.json").exists()
