import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ViscoelasticEVA:
    """
    Viscoelastic model for Ethylene Vinyl Acetate (EVA) encapsulant.
    Implements temperature- and frequency-dependent complex shear modulus:
        G*(f, T) = G'(f, T) + i * G''(f, T)
    where G' is the storage modulus, G'' is the loss modulus,
    and tan(delta) = G'' / G' is the loss factor.
    """
    density: float = 960.0             # kg/m^3
    thickness_m: float = 0.45e-3       # 0.45 mm
    g_relaxed: float = 1.0e6           # Relaxed rubbery shear modulus G_0 in Pa (1 MPa)
    g_unrelaxed: float = 80.0e6        # Unrelaxed glassy shear modulus G_inf in Pa (80 MPa)
    f_relax_ref: float = 100.0e3       # Characteristic relaxation frequency at T_ref (100 kHz)
    t_ref_k: float = 298.15            # Reference temperature 25 deg C (298.15 K)
    activation_energy_kj_mol: float = 55.0  # Apparent Arrhenius activation energy (kJ/mol)
    r_gas_constant: float = 8.314462   # J / (mol * K)

    def shift_factor_arrhenius(self, temp_c: float) -> float:
        """Compute Arrhenius temperature shift factor a_T = tau(T) / tau(T_ref)."""
        temp_k = temp_c + 273.15
        delta_inv_t = (1.0 / temp_k) - (1.0 / self.t_ref_k)
        exponent = (self.activation_energy_kj_mol * 1.0e3 / self.r_gas_constant) * delta_inv_t
        # Clamp exponent to prevent numerical overflow/underflow
        exponent = np.clip(exponent, -15.0, 15.0)
        return float(np.exp(exponent))

    def complex_shear_modulus(self, freq_hz: np.ndarray, temp_c: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute storage modulus G' and loss modulus G'' at given frequencies and temperature.
        Uses a fractional Zener / standard linear solid viscoelastic model.
        Returns:
            (G_prime, G_double_prime) in Pascals.
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        a_t = self.shift_factor_arrhenius(temp_c)
        omega = 2.0 * np.pi * freq_hz
        omega_eff = omega * a_t
        tau_relax = 1.0 / (2.0 * np.pi * self.f_relax_ref)
        wt = omega_eff * tau_relax

        # Standard Linear Solid (Zener) model
        delta_g = self.g_unrelaxed - self.g_relaxed
        denominator = 1.0 + wt**2
        g_prime = self.g_relaxed + delta_g * (wt**2) / denominator
        g_double_prime = delta_g * wt / denominator

        return g_prime, g_double_prime

    def attenuation_factor(self, freq_hz: np.ndarray, temp_c: float = 25.0) -> np.ndarray:
        """
        Acoustic waveguide attenuation coefficient alpha(f, T) in Np/m.
        In PV modules, the wave travels primarily in the 3.2 mm front glass waveguide (c_S0 ~ 5300 m/s),
        with viscoelastic boundary shear loss to the 0.45 mm EVA encapsulant.
        alpha ≈ (omega / c_glass) * (G'' / E_glass) * (h_eva / h_glass) * coupling_factor.
        Typically 0.5 - 3.0 Np/m (5 - 25 dB/m) in the 50-400 kHz AE band.
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        g_prime, g_double_prime = self.complex_shear_modulus(freq_hz, temp_c)
        omega = 2.0 * np.pi * freq_hz
        c_glass = 5300.0 # dominant Lamb S0 waveguide speed
        e_glass = 72.0e9 # Glass Young's modulus
        
        # Viscoelastic energy leakage fraction into EVA layer
        leakage_ratio = (g_double_prime / e_glass) * (self.thickness_m / 3.2e-3)
        alpha = (omega / c_glass) * leakage_ratio * 300.0 + 0.10 # baseline glass scattering
        return np.clip(alpha, 0.05, 5.0)
