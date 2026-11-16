"""
Semi-Analytical Finite Element (SAFE) solver for guided waves in layered plates.

The plate thickness z is discretised with 1D linear elements; the in-plane field
varies as exp(i (k x - omega t)).  For a given real wavenumber k the generalized
Hermitian eigenproblem

    (K1 + i k K2 + k^2 K3) U = omega^2 M U

is solved for the sagittal-plane (Lamb-type, u_x / u_z) modes.

History: before 2026-11 the cross-coupling block ``i k K2`` had two entries with the
wrong sign and the assembled matrix was then force-symmetrised, which silently
cancelled the inter-node coupling.  The resulting dispersion curves were unphysical
(S0 faster than the plate-wave limit, A0 several times too fast).  The element matrix
is now derived directly from the strain energy and is Hermitian by construction; a
regression test pins single-layer SAFE to the exact Rayleigh-Lamb solution.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .viscoelastic_eva import ViscoelasticEVA

Layer = Tuple[str, float, float, float, float]  # (name, thickness_m, density, E_pa, nu)


@dataclass
class ModeSolution:
    """Eigen-solution of the SAFE problem at one wavenumber."""

    k: float
    freqs_hz: np.ndarray          # (n_modes,)
    shapes: np.ndarray            # (n_dof, n_modes), mass-normalised (U^H M U = 1)
    inplane_fraction: np.ndarray  # kinetic-energy fraction carried by u_x, per mode


def default_pv_layers(eva_temp_c: float = 25.0, eva_ref_freq_hz: float = 150.0e3) -> List[Layer]:
    """Standard glass/EVA/Si/EVA/backsheet stack (EVA modulus from the viscoelastic model)."""
    eva = ViscoelasticEVA()
    g_eva, _ = eva.complex_shear_modulus(eva_ref_freq_hz, temp_c=eva_temp_c)
    nu_eva = 0.45
    e_eva = float(2.0 * g_eva * (1.0 + nu_eva))
    return [
        ("glass", 3.2e-3, 2500.0, 72.0e9, 0.22),
        ("eva_front", 0.45e-3, 960.0, e_eva, nu_eva),
        ("si_cell", 0.175e-3, 2330.0, 130.0e9, 0.28),
        ("eva_back", 0.45e-3, 960.0, e_eva, nu_eva),
        ("backsheet", 0.30e-3, 1200.0, 3.5e9, 0.35),
    ]


def _element_matrices(k: float, dz: float, rho: float, E: float, nu: float, shear_only: bool = False):
    """Hermitian element stiffness (4x4) and mass (4x4) for a linear 2-node element.

    DOF order: [u_x1, u_z1, u_x2, u_z2].  Strains for u = U(z) exp(i k x):
        eps_xx = i k u_x,  eps_zz = u_z',  gamma_xz = u_x' + i k u_z
    ``shear_only`` drops the Lamé-lambda (volumetric) part; used to compute the shear
    strain energy, which is the lossy part for polymer interlayers.
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = 0.0 if shear_only else (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c11 = lam + 2.0 * mu
    c13 = lam
    c33 = lam + 2.0 * mu
    c55 = mu

    me = (rho * dz / 6.0) * np.array(
        [[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]
    )
    ke_k2 = (k**2) * (dz / 6.0) * np.array(
        [
            [2 * c11, 0, c11, 0],
            [0, 2 * c55, 0, c55],
            [c11, 0, 2 * c11, 0],
            [0, c55, 0, 2 * c55],
        ],
        dtype=np.complex128,
    )
    ke_z = (1.0 / dz) * np.array(
        [[c55, 0, -c55, 0], [0, c33, 0, -c33], [-c55, 0, c55, 0], [0, -c33, 0, c33]],
        dtype=np.complex128,
    )
    a = c13 - c55
    b = c13 + c55
    ke_cross = 0.5j * k * np.array(
        [[0, a, 0, -b], [-a, 0, -b, 0], [0, b, 0, -a], [b, 0, a, 0]], dtype=np.complex128
    )
    ke = ke_k2 + ke_z + ke_cross
    return ke, me


class MultilayerSAFESolver:
    """SAFE solver for laminated plates (default: Glass / EVA / Silicon / EVA / Backsheet)."""

    def __init__(
        self,
        layers: Optional[Sequence[Layer]] = None,
        eva_temp_c: float = 25.0,
        n_elem_per_layer: int = 4,
    ):
        self.layers: List[Layer] = list(layers) if layers is not None else default_pv_layers(eva_temp_c)
        self.n_elem_per_layer = n_elem_per_layer
        self.total_thickness = sum(layer[1] for layer in self.layers)
        self.total_elements = len(self.layers) * n_elem_per_layer
        self.total_nodes = self.total_elements + 1
        # Node index range [first, last] belonging to each layer
        self.layer_nodes = [
            (i * n_elem_per_layer, (i + 1) * n_elem_per_layer) for i in range(len(self.layers))
        ]

    # ------------------------------------------------------------------ assembly
    def assemble_matrices(
        self, k: float, layer_index: Optional[int] = None, shear_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble global K(k) and M.  If ``layer_index`` is given, only that layer's
        contribution is assembled (used for modal strain-energy fractions)."""
        n_dof = 2 * self.total_nodes
        K = np.zeros((n_dof, n_dof), dtype=np.complex128)
        M = np.zeros((n_dof, n_dof), dtype=np.float64)
        node_idx = 0
        for li, (_name, thick, rho, E, nu) in enumerate(self.layers):
            dz = thick / self.n_elem_per_layer
            ke, me = _element_matrices(k, dz, rho, E, nu, shear_only=shear_only)
            for _ in range(self.n_elem_per_layer):
                if layer_index is None or layer_index == li:
                    dofs = [2 * node_idx, 2 * node_idx + 1, 2 * node_idx + 2, 2 * node_idx + 3]
                    K[np.ix_(dofs, dofs)] += ke
                    M[np.ix_(dofs, dofs)] += me
                node_idx += 1
        return K, M

    # ------------------------------------------------------------------ solving
    def solve(self, k: float, n_modes: int = 6) -> ModeSolution:
        """Lowest ``n_modes`` modes at wavenumber k with mass-normalised shapes."""
        K, M = self.assemble_matrices(k)
        from scipy.linalg import eigh

        w2, vecs = eigh(K, M)
        keep = w2 > 0
        w2 = w2[keep][:n_modes]
        vecs = vecs[:, keep][:, :n_modes]
        # mass-normalise
        for m in range(vecs.shape[1]):
            norm = np.sqrt(np.real(vecs[:, m].conj() @ M @ vecs[:, m]))
            vecs[:, m] /= max(norm, 1e-30)
        ux = vecs[0::2, :]
        uz = vecs[1::2, :]
        # kinetic energy fraction in u_x (lumped approximation is adequate for classification)
        ex = np.sum(np.abs(ux) ** 2, axis=0)
        ez = np.sum(np.abs(uz) ** 2, axis=0)
        frac = ex / np.maximum(ex + ez, 1e-30)
        return ModeSolution(k=k, freqs_hz=np.sqrt(w2) / (2.0 * np.pi), shapes=vecs, inplane_fraction=frac)

    def solve_modes_at_wavenumber(self, k: float, n_modes: int = 4) -> np.ndarray:
        """Backward-compatible: lowest natural frequencies (Hz) at wavenumber k."""
        return self.solve(k, n_modes=n_modes).freqs_hz

    def _nodal_mass(self) -> np.ndarray:
        """Lumped nodal mass per unit area (kg/m^2) for energy bookkeeping."""
        m = np.zeros(self.total_nodes)
        node = 0
        for _name, thick, rho, _E, _nu in self.layers:
            dz = thick / self.n_elem_per_layer
            for _ in range(self.n_elem_per_layer):
                m[node] += 0.5 * rho * dz
                m[node + 1] += 0.5 * rho * dz
                node += 1
        return m

    def waveguide_energy(self, sol: ModeSolution, waveguide_layer: int = 0):
        """Per mode: (fraction of kinetic energy in the waveguide layer,
        in-plane fraction of the kinetic energy inside the waveguide layer)."""
        m = self._nodal_mass()
        lo, hi = self.layer_nodes[waveguide_layer]
        ux = np.abs(sol.shapes[0::2, :]) ** 2
        uz = np.abs(sol.shapes[1::2, :]) ** 2
        e_tot = (m[:, None] * (ux + uz)).sum(axis=0)
        e_wg = (m[lo : hi + 1, None] * (ux[lo : hi + 1] + uz[lo : hi + 1])).sum(axis=0)
        ex_wg = (m[lo : hi + 1, None] * ux[lo : hi + 1]).sum(axis=0)
        return e_wg / np.maximum(e_tot, 1e-30), ex_wg / np.maximum(e_wg, 1e-30)

    def plate_velocity_limit(self, waveguide_layer: int = 0) -> float:
        """Low-frequency S0 (plate) speed of the waveguide layer — an upper bound for S0."""
        _n, _t, rho, E, nu = self.layers[waveguide_layer]
        return float(np.sqrt(E / (rho * (1.0 - nu**2))))

    def identify_fundamental_modes(self, sol: ModeSolution, waveguide_layer: int = 0) -> Tuple[int, int]:
        """Return (index_A0, index_S0) of the waveguide-layer (front glass) modes.

        In a PV laminate the soft EVA lets the silicon and backsheet vibrate almost
        independently, producing many slow "soft-layer" modes *below* the glass modes.
        Frequency order is therefore not a valid mode label.  Modes are classified by
        where their kinetic energy sits: glass-dominated modes only, then A0 = the
        lowest out-of-plane-dominated one and S0 = the most in-plane one.
        For a single-layer plate this reduces to the classical A0/S0.
        """
        wg_frac, inplane = self.waveguide_energy(sol, waveguide_layer)
        idx = np.arange(len(sol.freqs_hz))
        cp = 2.0 * np.pi * sol.freqs_hz / max(sol.k, 1e-12)
        # Fundamental modes never exceed the plate speed; this excludes higher-order
        # modes near cutoff (very large cp) that can also be in-plane dominated.
        bounded = cp <= 1.03 * self.plate_velocity_limit(waveguide_layer)
        glass = idx[(wg_frac >= 0.5) & bounded]
        if len(glass) == 0:
            glass = idx[bounded][:2] if np.any(bounded) else idx[:2]
        flex = [i for i in glass if inplane[i] < 0.5]
        i_a0 = int(min(flex, key=lambda i: sol.freqs_hz[i])) if flex else int(glass[0])
        ext = [i for i in glass if i != i_a0 and inplane[i] >= 0.5]
        if ext:
            # Most glass-confined extensional mode; glass/silicon hybrids carry part of
            # their energy in the (faster) silicon layer.
            i_s0 = int(max(ext, key=lambda i: wg_frac[i]))
        else:
            rest = [i for i in glass if i != i_a0]
            i_s0 = int(max(rest, key=lambda i: inplane[i])) if rest else int(idx[np.argmax(inplane)])
        return i_a0, i_s0

    def strain_energy_fractions(self, sol: ModeSolution, mode: int, shear_only_layers=()) -> np.ndarray:
        """Fraction of the mode's total strain energy attributed to each layer.

        For layers listed in ``shear_only_layers`` only the shear (deviatoric) part is
        returned, so that ``fractions @ loss_factors`` gives the modal loss factor with
        lossless bulk stiffness in polymer layers.
        """
        u = sol.shapes[:, mode]
        K_all, _ = self.assemble_matrices(sol.k)
        total = max(float(np.real(u.conj() @ K_all @ u)), 1e-30)
        energies = []
        for li in range(len(self.layers)):
            K_l, _ = self.assemble_matrices(sol.k, layer_index=li, shear_only=li in shear_only_layers)
            energies.append(max(float(np.real(u.conj() @ K_l @ u)), 0.0))
        return np.asarray(energies) / total

    def compute_dispersion(self, k_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Frequencies and phase velocities of the fundamental S0 and A0 modes.

        Returns (f_s0, cp_s0, f_a0, cp_a0) — same ordering as the historical API.
        """
        f_s0, cp_s0, f_a0, cp_a0 = [], [], [], []
        for k in np.asarray(k_range, dtype=np.float64):
            sol = self.solve(k, n_modes=14)
            ia, is_ = self.identify_fundamental_modes(sol)
            f_a0.append(sol.freqs_hz[ia])
            f_s0.append(sol.freqs_hz[is_])
            cp_a0.append(2.0 * np.pi * sol.freqs_hz[ia] / k)
            cp_s0.append(2.0 * np.pi * sol.freqs_hz[is_] / k)
        return np.array(f_s0), np.array(cp_s0), np.array(f_a0), np.array(cp_a0)
