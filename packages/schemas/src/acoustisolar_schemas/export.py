import json
from pathlib import Path
from typing import Dict, Type
from pydantic import BaseModel
from .module import ModuleTwinConfig, LayerConfig, ModuleLayoutConfig
from .sensor import SensorArrayConfig, PZTDiscConfig
from .daq import DAQConfig, PreampConfig, ADCConfig, TriggerConfig
from .source import SourceEventConfig
from .event import EventRecord, TriggerFeatures, GroundTruthLabel, ProvenanceRecord

SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "ModuleTwinConfig": ModuleTwinConfig,
    "LayerConfig": LayerConfig,
    "ModuleLayoutConfig": ModuleLayoutConfig,
    "SensorArrayConfig": SensorArrayConfig,
    "PZTDiscConfig": PZTDiscConfig,
    "DAQConfig": DAQConfig,
    "PreampConfig": PreampConfig,
    "ADCConfig": ADCConfig,
    "TriggerConfig": TriggerConfig,
    "SourceEventConfig": SourceEventConfig,
    "EventRecord": EventRecord,
    "TriggerFeatures": TriggerFeatures,
    "GroundTruthLabel": GroundTruthLabel,
    "ProvenanceRecord": ProvenanceRecord,
}

def export_json_schemas(output_dir: Path) -> Dict[str, Path]:
    """Export all registered Pydantic v2 schemas to JSON Schema files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files = {}

    for name, model_cls in SCHEMA_REGISTRY.items():
        schema_json = model_cls.model_json_schema()
        file_path = output_dir / f"{name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(schema_json, f, indent=2)
        exported_files[name] = file_path

    return exported_files

