import numpy as np
from typing import Tuple, List
from .materials import IsotropicMaterial, GLASS_MATERIAL

class LambDispersionSolver:
    """
    Exact analytical Rayleigh-Lamb dispersion solver for an isotropic free elastic plate.
    Plate thickness d, half-thickness h = d/2.
    
    Rayleigh-Lamb frequency equations:
    Symmetric (S) modes:
        tan(q*h) / tan(p*h) + 4 * k^2 * p * q / (k^2 - q^2)^2 = 0
    Antisymmetric (A) modes:
        tan(p*h) / tan(q*h) + 4 * k^2 * p * q / (k^2 - q^2)^2 = 0
    where:
        p = sqrt(omega^2 / c_L^2 - k^2)
        q = sqrt(omega^2 / c_T^2 - k^2)
        k = omega / c_phase
    """

    def __init__(self, material: IsotropicMaterial = GLASS_MATERIAL, thickness_m: float = 3.2e-3):
        self.material = material
        self.thickness_m = thickness_m
        self.h = thickness_m / 2.0
        self.cL = material.c_longitudinal
        self.cT = material.c_transverse

    def rayleigh_lamb_symmetric(self, cp: float, omega: float) -> float:
        """Characteristic function for symmetric (S) modes."""
        k = omega / cp
        h = self.h
        
        # Check cutoffs for real/imaginary radicals
        arg_p = (omega / self.cL)**2 - k**2
        arg_q = (omega / self.cT)**2 - k**2
        
        # Handle complex radicals stably
        p = 1j * np.sqrt(-arg_p) if arg_p < 0 else np.sqrt(arg_p)
        q = 1j * np.sqrt(-arg_q) if arg_q < 0 else np.sqrt(arg_q)
        
        # Avoid division by zero
        denom = (k**2 - q**2)**2
        if np.abs(denom) < 1e-12:
            return 1e6
            
        tan_qh = np.tan(q * h)
        tan_ph = np.tan(p * h)
        if np.abs(tan_ph) < 1e-12:
            return 1e6
            
        val = tan_qh / tan_ph + (4.0 * (k**2) * p * q) / denom
        return float(np.real(val))

    def rayleigh_lamb_antisymmetric(self, cp: float, omega: float) -> float:
        """Characteristic function for antisymmetric (A) modes."""
        k = omega / cp
        h = self.h
        
        arg_p = (omega / self.cL)**2 - k**2
        arg_q = (omega / self.cT)**2 - k**2
        
        p = 1j * np.sqrt(-arg_p) if arg_p < 0 else np.sqrt(arg_p)
        q = 1j * np.sqrt(-arg_q) if arg_q < 0 else np.sqrt(arg_q)
        
        denom = (k**2 - q**2)**2
        if np.abs(denom) < 1e-12:
            return 1e6
            
        tan_ph = np.tan(p * h)
        tan_qh = np.tan(q * h)
        if np.abs(tan_qh) < 1e-12:
            return 1e6
            
        val = tan_ph / tan_qh + (4.0 * (k**2) * p * q) / denom
        return float(np.real(val))

    def find_phase_velocity_s0(self, freq_hz: float) -> float:
        """
        Compute phase velocity cp for fundamental symmetric mode S0 at frequency freq_hz.
        At f -> 0, cp -> c_plate = sqrt(E / (rho * (1 - nu^2))).
        """
        if freq_hz <= 10.0:
            return self.material.c_plate_s0_limit
            
        omega = 2.0 * np.pi * freq_hz
        c_plate = self.material.c_plate_s0_limit
        
        # S0 phase velocity starts at c_plate at low f*d and gradually decreases.
        # Note: c = c_T (bulk shear speed ~3436 m/s) is a branch singularity (q = 0),
        # so for S0 mode we search strictly above c_T (cp_min = 1.05 * c_T).
        cp_min = self.cT * 1.05
        cp_max = c_plate * 1.02
        
        # Prioritize root closest to c_plate
        cps = np.linspace(cp_max, cp_min, 150)
        vals = [self.rayleigh_lamb_symmetric(c, omega) for c in cps]
        
        best_cp = c_plate
        min_dist_to_cplate = float('inf')
        for i in range(len(cps) - 1):
            if np.sign(vals[i]) != np.sign(vals[i+1]) and abs(vals[i] - vals[i+1]) < 20.0:
                low, high = min(cps[i], cps[i+1]), max(cps[i], cps[i+1])
                for _ in range(30):
                    mid = 0.5 * (low + high)
                    f_mid = self.rayleigh_lamb_symmetric(mid, omega)
                    if abs(f_mid) < 1e-6:
                        break
                    if np.sign(f_mid) == np.sign(self.rayleigh_lamb_symmetric(low, omega)):
                        low = mid
                    else:
                        high = mid
                cand = 0.5 * (low + high)
                dist = abs(cand - c_plate)
                if dist < min_dist_to_cplate:
                    min_dist_to_cplate = dist
                    best_cp = cand

        return float(best_cp)

    def find_phase_velocity_a0(self, freq_hz: float) -> float:
        """
        Compute phase velocity cp for fundamental antisymmetric mode A0 at frequency freq_hz.
        At low frequencies, A0 is highly dispersive with cp proportional to sqrt(omega).
        """
        if freq_hz <= 10.0:
            return 100.0
            
        omega = 2.0 * np.pi * freq_hz
        # Analytical thin plate flexural wave estimate as initial anchor:
        # cp_flexural = (D / (rho * d))^(1/4) * sqrt(omega)
        d = self.thickness_m
        E = self.material.youngs_modulus
        rho = self.material.density
        nu = self.material.poissons_ratio
        D = (E * d**3) / (12.0 * (1.0 - nu**2))
        cp_est = ((D / (rho * d))**0.25) * np.sqrt(omega)
        
        cp_min = max(cp_est * 0.5, 100.0)
        cp_max = min(max(cp_est * 1.5, self.cT * 0.95), self.cT * 1.05)
        
        cps = np.linspace(cp_min, cp_max, 150)
        vals = [self.rayleigh_lamb_antisymmetric(c, omega) for c in cps]
        
        best_cp = cp_est
        min_err = float('inf')
        for i in range(len(cps) - 1):
            if np.sign(vals[i]) != np.sign(vals[i+1]) and abs(vals[i] - vals[i+1]) < 50.0:
                low, high = cps[i], cps[i+1]
                for _ in range(30):
                    mid = 0.5 * (low + high)
                    f_mid = self.rayleigh_lamb_antisymmetric(mid, omega)
                    if abs(f_mid) < 1e-6:
                        break
                    if np.sign(f_mid) == np.sign(vals[i]):
                        low = mid
                    else:
                        high = mid
                cand = 0.5 * (low + high)
                err = abs(self.rayleigh_lamb_antisymmetric(cand, omega))
                if err < min_err:
                    min_err = err
                    best_cp = cand

        return float(best_cp)

    def compute_dispersion_curve(
        self, freq_range_hz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute phase and group velocities for S0 and A0 across frequencies.
        Returns:
            (cp_s0, cg_s0, cp_a0, cg_a0)
        Group velocity cg = d(omega) / d(k) = cp / (1 - (omega / cp) * (d cp / d omega)).
        """
        freqs = np.asarray(freq_range_hz, dtype=np.float64)
        cp_s0 = np.array([self.find_phase_velocity_s0(f) for f in freqs])
        cp_a0 = np.array([self.find_phase_velocity_a0(f) for f in freqs])
        
        # Numerical group velocities cg = d_omega / d_k
        omega = 2.0 * np.pi * freqs
        k_s0 = omega / cp_s0
        k_a0 = omega / cp_a0
        
        cg_s0 = np.gradient(omega, k_s0)
        cg_a0 = np.gradient(omega, k_a0)
        
        # Clean any numerical spikes at endpoints
        cg_s0 = np.clip(cg_s0, 0.5 * self.cT, 1.2 * self.material.c_plate_s0_limit)
        cg_a0 = np.clip(cg_a0, 100.0, 1.2 * self.cT)
        
        return cp_s0, cg_s0, cp_a0, cg_a0
