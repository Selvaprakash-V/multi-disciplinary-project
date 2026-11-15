import numpy as np
from typing import List, Tuple
from .materials import IsotropicMaterial, GLASS_MATERIAL, SILICON_MATERIAL, BACKSHEET_MATERIAL
from .viscoelastic_eva import ViscoelasticEVA

class MultilayerSAFESolver:
    """
    Semi-Analytical Finite Element (SAFE) solver for guided elastic waves
    in laminated multi-layered plates (Glass / EVA / Silicon / EVA / Backsheet).
    
    Discretizes the plate thickness z in [0, H] using 1D finite elements with
    in-plane wave propagation exp(i * (k * x - omega * t)).
    
    For a given wavenumber k, SAFE forms the generalized eigenvalue problem:
        (K(k) - omega^2 * M) * U = 0
    where K(k) = K1 + i * k * K2 + k^2 * K3.
    """

    def __init__(
        self,
        layers: List[Tuple[str, float, float, float, float]] = None,
        eva_temp_c: float = 25.0,
        n_elem_per_layer: int = 4
    ):
        """
        layers: list of (name, thickness_m, density_kg_m3, E_pa, nu)
        """
        if layers is None:
            eva = ViscoelasticEVA()
            g_eva, _ = eva.complex_shear_modulus(100.0e3, temp_c=eva_temp_c)
            # Estimate effective E from G: E = 2 * G * (1 + nu)
            nu_eva = 0.45
            e_eva = 2.0 * g_eva * (1.0 + nu_eva)
            
            self.layers = [
                ("glass", 3.2e-3, 2500.0, 72.0e9, 0.22),
                ("eva_front", 0.45e-3, 960.0, float(e_eva), nu_eva),
                ("si_cell", 0.175e-3, 2330.0, 130.0e9, 0.28),
                ("eva_back", 0.45e-3, 960.0, float(e_eva), nu_eva),
                ("backsheet", 0.30e-3, 1200.0, 3.5e9, 0.35),
            ]
        else:
            self.layers = layers
            
        self.n_elem_per_layer = n_elem_per_layer
        self.total_thickness = sum(layer[1] for layer in self.layers)
        self.total_elements = len(self.layers) * n_elem_per_layer
        self.total_nodes = self.total_elements + 1

    def assemble_matrices(self, k: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble global stiffness K(k) and mass M matrices for wavenumber k.
        Using 2 displacement degrees of freedom per node (u_x, u_z) for sagittal plane (P-SV / Lamb waves).
        """
        n_dof = 2 * self.total_nodes
        K = np.zeros((n_dof, n_dof), dtype=np.complex128)
        M = np.zeros((n_dof, n_dof), dtype=np.float64)
        
        node_idx = 0
        for name, thick, rho, E, nu in self.layers:
            dz = thick / self.n_elem_per_layer
            mu = E / (2.0 * (1.0 + nu))
            lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
            c11 = lam + 2.0 * mu
            c13 = lam
            c33 = lam + 2.0 * mu
            c55 = mu
            
            # Element mass matrix (2 nodes per element, linear shape functions)
            me = (rho * dz / 6.0) * np.array([
                [2.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0, 1.0],
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 2.0]
            ])
            
            # Analytical element stiffness integration for SAFE
            # B = L_x * d/dx + L_z * d/dz
            # For linear element shape functions N1 = 1 - xi, N2 = xi where xi = z / dz
            ke = np.zeros((4, 4), dtype=np.complex128)
            
            # k^2 terms (L_x^T * C * L_x)
            ke_k2 = (dz / 6.0) * np.array([
                [2*c11*k**2, 0, c11*k**2, 0],
                [0, 2*c55*k**2, 0, c55*k**2],
                [c11*k**2, 0, 2*c11*k**2, 0],
                [0, c55*k**2, 0, 2*c55*k**2]
            ], dtype=np.complex128)
            
            # d/dz terms (L_z^T * C * L_z)
            ke_z = (1.0 / dz) * np.array([
                [c55, 0, -c55, 0],
                [0, c33, 0, -c33],
                [-c55, 0, c55, 0],
                [0, -c33, 0, c33]
            ], dtype=np.complex128)
            
            # Cross terms i*k (L_x^T * C * L_z - L_z^T * C * L_x)
            ke_cross = 0.5 * 1j * k * np.array([
                [0, c13 - c55, 0, -(c13 + c55)],
                [c55 - c13, 0, c55 + c13, 0],
                [0, c13 + c55, 0, c55 - c13],
                [-(c55 + c13), 0, -(c55 - c13), 0]
            ], dtype=np.complex128)
            
            ke = ke_k2 + ke_z + ke_cross
            
            # Assemble into global matrices
            for elem in range(self.n_elem_per_layer):
                dofs = [
                    2 * node_idx, 2 * node_idx + 1,
                    2 * (node_idx + 1), 2 * (node_idx + 1) + 1
                ]
                for i in range(4):
                    for j in range(4):
                        K[dofs[i], dofs[j]] += ke[i, j]
                        M[dofs[i], dofs[j]] += me[i, j]
                node_idx += 1
                
        # Enforce Hermitian symmetry on K to ensure real positive eigenvalues for propagating modes
        K_herm = 0.5 * (K + K.conj().T)
        return K_herm, M

    def solve_modes_at_wavenumber(self, k: float, n_modes: int = 4) -> np.ndarray:
        """
        Solve for the lowest natural angular frequencies omega at wavenumber k.
        Returns sorted frequencies in Hz: freq_hz = omega / (2 * pi).
        """
        K, M = self.assemble_matrices(k)
        try:
            from scipy.linalg import eigh
            eigvals = eigh(K, M, eigvals_only=True)
        except ImportError:
            eigvals = np.linalg.eigvalsh(np.linalg.solve(M, K))
            
        valid_eigvals = np.real(eigvals[eigvals > 0])
        valid_eigvals.sort()
        omegas = np.sqrt(valid_eigvals[:n_modes])
        return omegas / (2.0 * np.pi)

    def compute_dispersion(self, k_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute modal frequencies, phase velocities (cp = omega / k) for fundamental modes.
        Returns:
            (k_values, freqs_mode0, cp_mode0, freqs_mode1, cp_mode1)
        """
        k_values = np.asarray(k_range, dtype=np.float64)
        f_s0 = []
        f_a0 = []
        cp_s0 = []
        cp_a0 = []
        
        for k in k_values:
            freqs = self.solve_modes_at_wavenumber(k, n_modes=2)
            if len(freqs) >= 2:
                # Mode 0 is flexural (A0-like), Mode 1 is extensional (S0-like) at low k
                f_a, f_s = freqs[0], freqs[1]
                f_a0.append(f_a)
                f_s0.append(f_s)
                cp_a0.append(2.0 * np.pi * f_a / k)
                cp_s0.append(2.0 * np.pi * f_s / k)
            else:
                f_a0.append(0.0)
                f_s0.append(0.0)
                cp_a0.append(0.0)
                cp_s0.append(0.0)
                
        return np.array(f_s0), np.array(cp_s0), np.array(f_a0), np.array(cp_a0)

