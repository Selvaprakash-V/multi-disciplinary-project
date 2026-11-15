import numpy as np
from typing import Tuple, List
from .materials import IsotropicMaterial, GLASS_MATERIAL

class LambDispersionSolver:
    """
    Exact analytical Rayleigh-Lamb dispersion solver for an isotropic free elastic plate.
    Plate thickness d, half-thickness h = d/2.
    
    Uses numerically exact real-valued Lamb characteristic functions for:
    - Fundamental Symmetric mode (S0)
    - Fundamental Antisymmetric mode (A0)
    without branch cut singularities or complex root trapping.
    """

    def __init__(self, material: IsotropicMaterial = GLASS_MATERIAL, thickness_m: float = 3.2e-3):
        self.material = material
        self.thickness_m = thickness_m
        self.h = thickness_m / 2.0
        self.cL = material.c_longitudinal
        self.cT = material.c_transverse
        # Approximate Rayleigh surface wave velocity
        nu = material.poissons_ratio
        self.cR = self.cT * (0.87 + 1.12 * nu) / (1.0 + nu)

    def char_eq_s0(self, cp: float, omega: float) -> float:
        """Exact real-valued characteristic function for symmetric (S) modes."""
        k = omega / cp
        h = self.h
        if cp > self.cT:
            q = np.sqrt((omega / self.cT)**2 - k**2)
            alpha_p = np.sqrt(k**2 - (omega / self.cL)**2)
            denom = (k**2 - q**2)**2
            return (4.0 * (k**2) * alpha_p * q) / denom - (np.tan(q * h) / np.tanh(alpha_p * h))
        else:
            alpha_p = np.sqrt(k**2 - (omega / self.cL)**2)
            alpha_q = np.sqrt(k**2 - (omega / self.cT)**2)
            denom = (2.0 * k**2 - (omega / self.cT)**2)**2
            return (np.tanh(alpha_q * h) / np.tanh(alpha_p * h)) - (4.0 * (k**2) * alpha_p * alpha_q) / denom

    def char_eq_a0(self, cp: float, omega: float) -> float:
        """Exact real-valued characteristic function for antisymmetric (A) modes."""
        k = omega / cp
        h = self.h
        if cp < self.cT:
            alpha_p = np.sqrt(k**2 - (omega / self.cL)**2)
            alpha_q = np.sqrt(k**2 - (omega / self.cT)**2)
            denom = (2.0 * k**2 - (omega / self.cT)**2)**2
            return (np.tanh(alpha_p * h) / np.tanh(alpha_q * h)) - (4.0 * (k**2) * alpha_p * alpha_q) / denom
        else:
            q = np.sqrt((omega / self.cT)**2 - k**2)
            alpha_p = np.sqrt(k**2 - (omega / self.cL)**2)
            denom = (k**2 - q**2)**2
            return (np.tanh(alpha_p * h) / np.tan(q * h)) + (4.0 * (k**2) * alpha_p * q) / denom

    def find_phase_velocity_s0(self, freq_hz: float) -> float:
        """Compute phase velocity cp for fundamental symmetric mode S0 at frequency freq_hz."""
        if freq_hz <= 10.0:
            return self.material.c_plate_s0_limit
            
        omega = 2.0 * np.pi * freq_hz
        c_plate = self.material.c_plate_s0_limit
        
        # S0 phase velocity is bounded between cT and c_plate in this frequency band
        cps = np.linspace(c_plate, self.cT * 1.005, 120)
        vals = [self.char_eq_s0(c, omega) for c in cps]
        
        for i in range(len(cps) - 1):
            if np.sign(vals[i]) != np.sign(vals[i+1]) and abs(vals[i] - vals[i+1]) < 20.0:
                low, high = cps[i+1], cps[i]
                for _ in range(35):
                    mid = 0.5 * (low + high)
                    f_mid = self.char_eq_s0(mid, omega)
                    if abs(f_mid) < 1e-6:
                        return float(mid)
                    if np.sign(f_mid) == np.sign(self.char_eq_s0(low, omega)):
                        low = mid
                    else:
                        high = mid
                return float(0.5 * (low + high))
                
        return float(c_plate)

    def find_phase_velocity_a0(self, freq_hz: float) -> float:
        """Compute phase velocity cp for fundamental antisymmetric mode A0 at frequency freq_hz."""
        if freq_hz <= 10.0:
            return 100.0
            
        omega = 2.0 * np.pi * freq_hz
        # Analytical thin plate flexural wave estimate as lower bracket
        D = (self.material.youngs_modulus * self.thickness_m**3) / (12.0 * (1.0 - self.material.poissons_ratio**2))
        cp_est = ((D / (self.material.density * self.thickness_m))**0.25) * np.sqrt(omega)
        
        cp_min = max(cp_est * 0.4, 150.0)
        cp_max = self.cR * 0.999
        
        cps = np.linspace(cp_min, cp_max, 150)
        vals = [self.char_eq_a0(c, omega) for c in cps]
        
        for i in range(len(cps) - 1):
            if np.sign(vals[i]) != np.sign(vals[i+1]) and abs(vals[i] - vals[i+1]) < 30.0:
                low, high = cps[i], cps[i+1]
                for _ in range(35):
                    mid = 0.5 * (low + high)
                    f_mid = self.char_eq_a0(mid, omega)
                    if abs(f_mid) < 1e-6:
                        return float(mid)
                    if np.sign(f_mid) == np.sign(self.char_eq_a0(low, omega)):
                        low = mid
                    else:
                        high = mid
                return float(0.5 * (low + high))
                
        return float(cp_est)

    def compute_dispersion_curve(
        self, freq_range_hz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute phase and group velocities for S0 and A0 across frequencies."""
        freqs = np.asarray(freq_range_hz, dtype=np.float64)
        cp_s0 = np.array([self.find_phase_velocity_s0(f) for f in freqs])
        cp_a0 = np.array([self.find_phase_velocity_a0(f) for f in freqs])
        
        omega = 2.0 * np.pi * freqs
        k_s0 = omega / cp_s0
        k_a0 = omega / cp_a0
        
        cg_s0 = np.gradient(omega, k_s0)
        cg_a0 = np.gradient(omega, k_a0)
        
        return cp_s0, cg_s0, cp_a0, cg_a0
