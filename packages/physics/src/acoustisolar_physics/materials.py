import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class IsotropicMaterial:
    """Isotropic elastic material properties and acoustic wave speeds."""
    name: str
    density: float         # kg / m^3
    youngs_modulus: float  # Pa
    poissons_ratio: float  # dimensionless

    @property
    def bulk_modulus(self) -> float:
        return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))

    @property
    def shear_modulus(self) -> float:
        """Lamé parameter mu."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))

    @property
    def lame_lambda(self) -> float:
        """Lamé parameter lambda."""
        E = self.youngs_modulus
        nu = self.poissons_ratio
        return (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def c_longitudinal(self) -> float:
        """Bulk longitudinal (P) wave speed in m/s."""
        return np.sqrt((self.lame_lambda + 2.0 * self.shear_modulus) / self.density)

    @property
    def c_transverse(self) -> float:
        """Bulk transverse / shear (S) wave speed in m/s."""
        return np.sqrt(self.shear_modulus / self.density)

    @property
    def c_plate_s0_limit(self) -> float:
        """Theoretical low-frequency fundamental symmetric (S0) plate mode speed in m/s."""
        E = self.youngs_modulus
        rho = self.density
        nu = self.poissons_ratio
        return np.sqrt(E / (rho * (1.0 - nu**2)))

# Standard material presets for solar modules
GLASS_MATERIAL = IsotropicMaterial(
    name="tempered_solar_glass",
    density=2500.0,
    youngs_modulus=72.0e9,
    poissons_ratio=0.22,
)

SILICON_MATERIAL = IsotropicMaterial(
    name="monocrystalline_silicon_effective",
    density=2330.0,
    youngs_modulus=130.0e9,
    poissons_ratio=0.28,
)

BACKSHEET_MATERIAL = IsotropicMaterial(
    name="pvdf_backsheet",
    density=1200.0,
    youngs_modulus=3.5e9,
    poissons_ratio=0.35,
)
