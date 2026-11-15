from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class LayerConfig(BaseModel):
    """Configuration for a single material layer in the PV laminate stack."""
    name: str = Field(..., description="Layer name (e.g. glass, eva_front, si_cell, eva_back, backsheet)")
    thickness_mm: float = Field(..., gt=0, description="Layer thickness in millimeters")
    density_kg_m3: float = Field(default=2500.0, gt=0, description="Material mass density in kg/m^3")
    youngs_modulus_gpa: float = Field(default=72.0, gt=0, description="Young's modulus in GPa")
    poissons_ratio: float = Field(default=0.22, ge=0.0, lt=0.5, description="Poisson's ratio")
    damping_ratio: float = Field(default=0.01, ge=0.0, description="Viscous/material loss factor")
    material_model: Literal["isotropic_elastic", "viscoelastic_prony", "anisotropic_silicon"] = "isotropic_elastic"
    temperature_c: float = Field(default=25.0, description="Layer temperature in Celsius")

class ModuleLayoutConfig(BaseModel):
    """Layout of the photovoltaic cells on the module."""
    cells: int = Field(default=120, gt=0, description="Total number of solar cells")
    half_cut: bool = Field(default=True, description="Whether cells are half-cut")
    cell_size: str = Field(default="M10", description="Cell wafer standard: M6, M10, G12")
    busbars: int = Field(default=9, ge=3, le=16, description="Number of busbars per cell")
    rows: int = Field(default=20, gt=0, description="Number of cell rows")
    cols: int = Field(default=6, gt=0, description="Number of cell columns")
    cell_width_mm: float = Field(default=91.0, gt=0, description="Cell width in mm")
    cell_height_mm: float = Field(default=182.0, gt=0, description="Cell height in mm")

class ModuleTwinConfig(BaseModel):
    """Full digital twin specification for a PV module."""
    module_id: str = Field(default="MOD-DEMO-001", description="Unique module identifier")
    name: str = Field(default="Standard 120 Half-Cut Mono PERC", description="Descriptive module name")
    width_mm: float = Field(default=1040.0, gt=0, description="Total module width in mm")
    length_mm: float = Field(default=1760.0, gt=0, description="Total module length in mm")
    layout: ModuleLayoutConfig = Field(default_factory=ModuleLayoutConfig)
    stack: List[LayerConfig] = Field(
        default_factory=lambda: [
            LayerConfig(name="glass", thickness_mm=3.2, density_kg_m3=2500.0, youngs_modulus_gpa=72.0, poissons_ratio=0.22),
            LayerConfig(name="eva_front", thickness_mm=0.45, density_kg_m3=960.0, youngs_modulus_gpa=0.0125, poissons_ratio=0.45, material_model="viscoelastic_prony"),
            LayerConfig(name="si_cell", thickness_mm=0.175, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28, material_model="anisotropic_silicon"),
            LayerConfig(name="eva_back", thickness_mm=0.45, density_kg_m3=960.0, youngs_modulus_gpa=0.0125, poissons_ratio=0.45, material_model="viscoelastic_prony"),
            LayerConfig(name="backsheet", thickness_mm=0.30, density_kg_m3=1200.0, youngs_modulus_gpa=3.5, poissons_ratio=0.35),
        ]
    )

    def total_thickness_mm(self) -> float:
        return sum(layer.thickness_mm for layer in self.stack)

