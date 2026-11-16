"""
Mode-resolved guided-wave model of a PV laminate, built on the SAFE solver.

For a configured layer stack it pre-computes, on a temperature grid, the properties of the
two fundamental Lamb-type modes (A0 flexural, S0 extensional) needed by the Tier-0
simulator:

* phase and group velocity           c_p(f), c_g(f)
* attenuation alpha(f) [Np/m]        from the modal strain-energy method:
                                       eta_mode = sum_l (U_l / U) * eta_l ,
                                       alpha    = eta_mode * omega / (2 c_g)
  where eta_l is the EVA loss factor tan(delta)(f, T) for viscoelastic layers and the
  configured ``damping_ratio`` for elastic layers.  The soft interlayer therefore damps
  A0 and S0 differently, as observed for laminated glass.
* coupling factors (mass-normalised mode shapes):
    - receiver: out-of-plane surface displacement |u_z| at the glass top (PZT discs are
      mainly out-of-plane sensitive)
    - in-plane dipole source in the silicon layer: k |u_x(Si)|          (cracks)
    - normal force on the glass surface: |u_z(top)|                      (rain, hail, taps)
    - in-plane force at mid-glass: |u_x(glass mid)|                      (frame stick-slip)

All material constants are [CALIBRATE]; the model is a physics-shaped prior, not a
validated transfer function.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .safe import MultilayerSAFESolver
from .viscoelastic_eva import ViscoelasticEVA

MODES = ("A0", "S0")
SOURCE_COUPLINGS = ("inplane_dipole_si", "normal_force_top", "inplane_force_glass")


@dataclass
class LayerSpec:
    name: str
    thickness_m: float
    density: float
    youngs_modulus_pa: float
    poissons_ratio: float
    loss_factor: float = 0.01
    viscoelastic: bool = False


def _median_smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Running median with edge padding (odd ``width``)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < width:
        return x
    h = width // 2
    xp = np.pad(x, h, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(xp, width)
    return np.median(windows, axis=-1)


def layers_from_module_config(module_config) -> List[LayerSpec]:
    """Convert a ``ModuleTwinConfig`` stack into ``LayerSpec`` objects."""
    specs = []
    for layer in module_config.stack:
        specs.append(
            LayerSpec(
                name=layer.name,
                thickness_m=layer.thickness_mm * 1.0e-3,
                density=layer.density_kg_m3,
                youngs_modulus_pa=layer.youngs_modulus_gpa * 1.0e9,
                poissons_ratio=layer.poissons_ratio,
                loss_factor=layer.damping_ratio,
                viscoelastic=(layer.material_model == "viscoelastic_prony"),
            )
        )
    return specs


@dataclass
class _ModeTable:
    f: np.ndarray
    cp: np.ndarray
    cg: np.ndarray
    alpha: np.ndarray
    receiver: np.ndarray
    source: Dict[str, np.ndarray] = field(default_factory=dict)


class LaminateDispersionModel:
    """Pre-computed A0/S0 properties of a laminate on a temperature grid."""

    def __init__(
        self,
        layers: Sequence[LayerSpec],
        temperatures_c: Sequence[float] = (-30.0, -10.0, 10.0, 30.0, 50.0, 70.0, 90.0),
        k_grid: Optional[np.ndarray] = None,
        n_elem_per_layer: int = 3,
        eva_model: Optional[ViscoelasticEVA] = None,
        eva_ref_freq_hz: float = 150.0e3,
    ):
        self.layers = list(layers)
        self.temperatures = np.asarray(sorted(temperatures_c), dtype=np.float64)
        self.k_grid = k_grid if k_grid is not None else np.logspace(0.0, np.log10(2500.0), 72)
        self.n_elem = n_elem_per_layer
        self.eva = eva_model or ViscoelasticEVA()
        self.eva_ref_freq_hz = eva_ref_freq_hz
        # Reference EVA modulus at 25 C so configured E values act as the 25 C anchor
        self._eva_e_ref = self.eva.youngs_modulus(eva_ref_freq_hz, 25.0, nu=0.45)
        self.si_index = self._find_layer(("si", "cell"))
        self.tables: List[Dict[str, _ModeTable]] = [self._compute(t) for t in self.temperatures]

    # ------------------------------------------------------------------ helpers
    def _find_layer(self, keys) -> int:
        for i, l in enumerate(self.layers):
            if any(k in l.name.lower() for k in keys):
                return i
        return min(2, len(self.layers) - 1)

    def _safe_layers(self, temp_c: float):
        out = []
        for l in self.layers:
            E = l.youngs_modulus_pa
            if l.viscoelastic:
                scale = l.youngs_modulus_pa / max(self._eva_e_ref, 1.0)
                E = scale * self.eva.youngs_modulus(self.eva_ref_freq_hz, temp_c, nu=l.poissons_ratio)
            out.append((l.name, l.thickness_m, l.density, float(E), l.poissons_ratio))
        return out

    def _layer_loss(self, freq_hz: float, temp_c: float) -> np.ndarray:
        eta = np.empty(len(self.layers))
        for i, l in enumerate(self.layers):
            eta[i] = float(self.eva.loss_factor(freq_hz, temp_c)) if l.viscoelastic else l.loss_factor
        return eta

    def _compute(self, temp_c: float) -> Dict[str, _ModeTable]:
        solver = MultilayerSAFESolver(layers=self._safe_layers(temp_c), n_elem_per_layer=self.n_elem)
        n_nodes = solver.total_nodes
        si_lo, si_hi = solver.layer_nodes[self.si_index]
        si_node = (si_lo + si_hi) // 2
        g_lo, g_hi = solver.layer_nodes[0]
        glass_mid = (g_lo + g_hi) // 2

        visco = tuple(i for i, l in enumerate(self.layers) if l.viscoelastic)
        raw = {m: {"f": [], "rcv": [], "dip": [], "nf": [], "ipf": [], "frac": []} for m in MODES}
        for k in self.k_grid:
            sol = solver.solve(float(k), n_modes=14)
            ia, is_ = solver.identify_fundamental_modes(sol)
            for name, idx in (("A0", ia), ("S0", is_)):
                u = sol.shapes[:, idx]
                ux = u[0::2]
                uz = u[1::2]
                r = raw[name]
                r["f"].append(sol.freqs_hz[idx])
                r["rcv"].append(abs(uz[0]))
                r["dip"].append(k * abs(ux[min(si_node, n_nodes - 1)]))
                r["nf"].append(abs(uz[0]))
                r["ipf"].append(abs(ux[glass_mid]))
                r["frac"].append(solver.strain_energy_fractions(sol, idx, shear_only_layers=visco))

        tables = {}
        for name in MODES:
            r = raw[name]
            f = np.asarray(r["f"])
            order = np.argsort(f)
            f = f[order]
            k = self.k_grid[order]
            omega = 2.0 * np.pi * f
            # Glass modes hybridise with soft-layer modes at avoided crossings, which makes
            # the per-k pick jump between branches.  Median-smooth before differentiating.
            cp = _median_smooth(omega / k, 5)
            omega_s = cp * k
            cg = np.gradient(omega_s, k)
            # Normal dispersion bounds: S0 cg <= cp; A0 cp <= cg <= 2 cp
            lo, hi = (0.5, 1.0) if name == "S0" else (1.0, 2.0)
            cg = np.clip(_median_smooth(cg, 5), lo * cp, hi * cp)
            fracs = np.asarray(r["frac"])[order]
            eta = np.array([fracs[i] @ self._layer_loss(f[i], temp_c) for i in range(len(f))])
            alpha = _median_smooth(eta * omega_s / (2.0 * cg), 5)
            f_s = omega_s / (2.0 * np.pi)
            rcv = _median_smooth(np.asarray(r["rcv"])[order], 5)
            src = {
                "inplane_dipole_si": _median_smooth(np.asarray(r["dip"])[order], 5),
                "normal_force_top": _median_smooth(np.asarray(r["nf"])[order], 5),
                "inplane_force_glass": _median_smooth(np.asarray(r["ipf"])[order], 5),
            }
            # Smoothing cp can make f = cp k locally non-monotonic; np.interp requires an
            # increasing abscissa (otherwise it silently returns wrong values), so re-sort.
            o2 = np.argsort(f_s, kind="stable")
            tables[name] = _ModeTable(
                f=f_s[o2],
                cp=cp[o2],
                cg=cg[o2],
                alpha=alpha[o2],
                receiver=rcv[o2],
                source={k: v[o2] for k, v in src.items()},
            )
        return tables

    # ------------------------------------------------------------------ queries
    def evaluate(self, freqs_hz: np.ndarray, temp_c: float) -> Dict[str, Dict[str, np.ndarray]]:
        """Interpolate all mode properties at ``freqs_hz`` and temperature ``temp_c``."""
        freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        t = float(np.clip(temp_c, self.temperatures[0], self.temperatures[-1]))
        j = int(np.clip(np.searchsorted(self.temperatures, t) - 1, 0, len(self.temperatures) - 2))
        t0, t1 = self.temperatures[j], self.temperatures[j + 1]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

        def interp(tab: _ModeTable, arr: np.ndarray) -> np.ndarray:
            return np.interp(freqs_hz, tab.f, arr)

        out = {}
        for name in MODES:
            a, b = self.tables[j][name], self.tables[j + 1][name]
            res = {}
            for key in ("cp", "cg", "alpha", "receiver"):
                res[key] = (1 - w) * interp(a, getattr(a, key)) + w * interp(b, getattr(b, key))
            for key in SOURCE_COUPLINGS:
                res[key] = (1 - w) * interp(a, a.source[key]) + w * interp(b, b.source[key])
            res["cp"] = self._causal_phase_velocity(freqs_hz, res["cp"], self.group_velocity_limit(name))
            out[name] = res
        return out

    def group_velocity_limit(self, mode: str) -> float:
        """Upper bound on group velocity: glass plate speed (S0) or glass shear speed (A0)."""
        g = self.layers[0]
        if mode == "S0":
            return 1.01 * float(np.sqrt(g.youngs_modulus_pa / (g.density * (1.0 - g.poissons_ratio**2))))
        shear = g.youngs_modulus_pa / (2.0 * (1.0 + g.poissons_ratio))
        return 1.05 * float(np.sqrt(shear / g.density))

    @staticmethod
    def _causal_phase_velocity(freqs_hz: np.ndarray, cp: np.ndarray, v_group_max: float) -> np.ndarray:
        """Make k(omega) = omega / c_p monotonic with group velocity <= ``v_group_max``.

        Tabulated curves have small jumps where glass modes hybridise with soft-layer
        modes; a locally decreasing k(omega) means negative group delay, i.e. energy
        arriving *before* the source fires.  A forward pass enforces
        dk/domega >= 1 / v_group_max, which guarantees causality.
        """
        order = np.argsort(freqs_hz)
        f = freqs_hz[order]
        omega = 2.0 * np.pi * f
        k = omega / np.maximum(cp[order], 1.0)
        min_dk = np.diff(omega) / v_group_max
        for i in range(1, len(k)):
            k[i] = max(k[i], k[i - 1] + min_dk[i - 1])
        cp_out = np.empty_like(cp)
        with np.errstate(divide="ignore", invalid="ignore"):
            cp_sorted = np.where(k > 0, omega / np.maximum(k, 1e-12), cp[order])
        cp_sorted[0] = cp[order][0]
        cp_out[order] = cp_sorted
        return cp_out

    def s0_velocity(self, temp_c: float, freq_hz: float = 150.0e3) -> float:
        """S0 phase velocity at a reference frequency (used for TDOA)."""
        return float(self.evaluate(np.array([freq_hz]), temp_c)["S0"]["cp"][0])
