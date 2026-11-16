from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ViscoelasticEVA:
    """
    Viscoelastic model for the Ethylene Vinyl Acetate (EVA) encapsulant.

    Complex shear modulus G*(f, T) = G'(f, T) + i G''(f, T) from a Zener (standard linear
    solid) model with a time-temperature shift factor a_T.

    Temperature shift (``shift_model``):
      * ``"arrhenius"`` – a_T = exp(Ea/R (1/T - 1/T_ref)); adequate well above T_g.
      * ``"hybrid"`` (default) – WLF below ``T_g + wlf_window_c`` and Arrhenius above,
        joined continuously.  Near the glass transition polymers follow WLF behaviour and
        Arrhenius underestimates the stiffening; this matters for winter operation.
    All constants are [CALIBRATE] — see ASSUMPTIONS.md §1.
    """

    density: float = 960.0             # kg/m^3
    thickness_m: float = 0.45e-3       # 0.45 mm
    g_relaxed: float = 1.0e6           # rubbery shear modulus G_0 (Pa)
    g_unrelaxed: float = 80.0e6        # glassy shear modulus G_inf (Pa)
    f_relax_ref: float = 100.0e3       # relaxation frequency at T_ref (Hz)  [CALIBRATE]
    t_ref_k: float = 298.15            # 25 C
    activation_energy_kj_mol: float = 55.0
    r_gas_constant: float = 8.314462
    t_glass_c: float = -35.0           # glass transition temperature
    wlf_c1: float = 17.44              # universal WLF constants referenced to T_g
    wlf_c2: float = 51.6
    wlf_window_c: float = 50.0         # WLF applies for T < T_g + window
    shift_model: str = "hybrid"

    # ------------------------------------------------------------------ shift factors
    def _log_at_arrhenius(self, temp_c: float) -> float:
        temp_k = temp_c + 273.15
        return (self.activation_energy_kj_mol * 1.0e3 / self.r_gas_constant) * (
            1.0 / temp_k - 1.0 / self.t_ref_k
        )

    def _log_at_wlf_tg(self, temp_c: float) -> float:
        """Natural-log WLF shift referenced to T_g."""
        dt = temp_c - self.t_glass_c
        dt = max(dt, -0.9 * self.wlf_c2)  # stay away from the WLF pole
        return -np.log(10.0) * self.wlf_c1 * dt / (self.wlf_c2 + dt)

    def shift_factor(self, temp_c: float) -> float:
        """Time-temperature shift factor a_T = tau(T) / tau(T_ref)."""
        if self.shift_model == "arrhenius":
            log_at = self._log_at_arrhenius(temp_c)
        else:
            t_switch = self.t_glass_c + self.wlf_window_c
            if temp_c >= t_switch:
                log_at = self._log_at_arrhenius(temp_c)
            else:
                log_at = self._log_at_arrhenius(t_switch) + (
                    self._log_at_wlf_tg(temp_c) - self._log_at_wlf_tg(t_switch)
                )
        return float(np.exp(np.clip(log_at, -30.0, 30.0)))

    def shift_factor_arrhenius(self, temp_c: float) -> float:
        """Backward-compatible pure-Arrhenius shift factor."""
        return float(np.exp(np.clip(self._log_at_arrhenius(temp_c), -15.0, 15.0)))

    # ------------------------------------------------------------------ moduli
    def complex_shear_modulus(self, freq_hz, temp_c: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
        """Storage modulus G' and loss modulus G'' (Pa) at the given frequencies."""
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        a_t = self.shift_factor(temp_c)
        omega_eff = 2.0 * np.pi * freq_hz * a_t
        tau_relax = 1.0 / (2.0 * np.pi * self.f_relax_ref)
        wt = omega_eff * tau_relax
        delta_g = self.g_unrelaxed - self.g_relaxed
        denominator = 1.0 + wt**2
        g_prime = self.g_relaxed + delta_g * (wt**2) / denominator
        g_double_prime = delta_g * wt / denominator
        return g_prime, g_double_prime

    def loss_factor(self, freq_hz, temp_c: float = 25.0) -> np.ndarray:
        """tan(delta) = G''/G'."""
        g_p, g_pp = self.complex_shear_modulus(freq_hz, temp_c)
        return g_pp / np.maximum(g_p, 1e-12)

    def youngs_modulus(self, freq_hz: float, temp_c: float = 25.0, nu: float = 0.45) -> float:
        """Effective storage Young's modulus E' = 2 G' (1 + nu)."""
        g_p, _ = self.complex_shear_modulus(freq_hz, temp_c)
        return float(2.0 * np.asarray(g_p) * (1.0 + nu))

    def attenuation_factor(self, freq_hz, temp_c: float = 25.0) -> np.ndarray:
        """
        Legacy single-mode attenuation estimate alpha(f, T) in Np/m.

        Superseded by the mode-resolved modal-strain-energy attenuation computed in
        ``LaminateDispersionModel`` (A0 and S0 are damped differently by the soft
        interlayer).  Kept for backward compatibility and quick estimates; the scale
        constant is an engineering fit, not a sourced value.
        """
        freq_hz = np.asarray(freq_hz, dtype=np.float64)
        _g_prime, g_double_prime = self.complex_shear_modulus(freq_hz, temp_c)
        omega = 2.0 * np.pi * freq_hz
        c_glass = 5300.0
        e_glass = 72.0e9
        leakage_ratio = (g_double_prime / e_glass) * (self.thickness_m / 3.2e-3)
        alpha = (omega / c_glass) * leakage_ratio * 300.0 + 0.10
        return np.clip(alpha, 0.05, 5.0)
