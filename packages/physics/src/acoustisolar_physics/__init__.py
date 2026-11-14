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
from .source_library import SourceLibrary
from .tier0 import Tier0Simulator
from .tier1 import Tier1WaveSolver

__all__ = [
    "IsotropicMaterial",
    "GLASS_MATERIAL",
    "SILICON_MATERIAL",
    "BACKSHEET_MATERIAL",
    "ViscoelasticEVA",
    "LambDispersionSolver",
    "MultilayerSAFESolver",
    "SourceLibrary",
    "Tier0Simulator",
    "Tier1WaveSolver",
]
