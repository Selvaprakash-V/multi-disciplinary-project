"""AcoustiSolar Schemas: Pydantic v2 domain models for digital twins, DAQ, and events."""

from .module import ModuleTwinConfig, LayerConfig, ModuleLayoutConfig
from .sensor import SensorArrayConfig, PZTDiscConfig
from .daq import DAQConfig, PreampConfig, ADCConfig, TriggerConfig
from .source import SourceClass, CrackType, NoiseClass, SourceEventConfig
from .event import EventRecord, TriggerFeatures, GroundTruthLabel, ProvenanceRecord
from .hashing import compute_config_hash, canonical_json_dump
from .export import export_json_schemas, SCHEMA_REGISTRY

__all__ = [
    "ModuleTwinConfig",
    "LayerConfig",
    "ModuleLayoutConfig",
    "SensorArrayConfig",
    "PZTDiscConfig",
    "DAQConfig",
    "PreampConfig",
    "ADCConfig",
    "TriggerConfig",
    "SourceClass",
    "CrackType",
    "NoiseClass",
    "SourceEventConfig",
    "EventRecord",
    "TriggerFeatures",
    "GroundTruthLabel",
    "ProvenanceRecord",
    "compute_config_hash",
    "canonical_json_dump",
    "export_json_schemas",
    "SCHEMA_REGISTRY",
]

