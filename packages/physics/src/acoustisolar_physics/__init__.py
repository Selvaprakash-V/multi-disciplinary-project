"""AcoustiSolar Physics: Guided Lamb-wave, SAFE dispersion, and elastodynamic wave propagation."""

from .materials import (
    IsotropicMaterial,
    GLASS_MATERIAL,
    SILICON_MATERIAL,
    BACKSHEET_MATERIAL,
)
from .viscoelastic_eva import ViscoelasticEVA
from .lamb import LambDispersionSolver
from .safe import MultilayerSAFESolver
from .laminate import LaminateDispersionModel, LayerSpec, layers_from_module_config
from .source_library import SourceLibrary
from .tier0 import Tier0Simulator, laminate_for_module
from .tier1 import Tier1WaveSolver

__all__ = [
    "IsotropicMaterial",
    "GLASS_MATERIAL",
    "SILICON_MATERIAL",
    "BACKSHEET_MATERIAL",
    "ViscoelasticEVA",
    "LambDispersionSolver",
    "MultilayerSAFESolver",
    "LaminateDispersionModel",
    "LayerSpec",
    "layers_from_module_config",
    "SourceLibrary",
    "Tier0Simulator",
    "laminate_for_module",
    "Tier1WaveSolver",
]
