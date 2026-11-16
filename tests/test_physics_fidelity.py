"""Regression tests for the physics-fidelity fixes (docs/research/CODEBASE_EVALUATION.md §2-3)."""

import numpy as np
import pytest

from acoustisolar_physics.lamb import LambDispersionSolver
from acoustisolar_physics.safe import MultilayerSAFESolver
from acoustisolar_physics.tier0 import Tier0Simulator, laminate_for_module
from acoustisolar_physics.tier1 import Tier1WaveSolver
from acoustisolar_physics.viscoelastic_eva import ViscoelasticEVA
from acoustisolar_schemas.module import ModuleTwinConfig
from acoustisolar_schemas.sensor import SensorArrayConfig
from acoustisolar_schemas.source import NoiseClass, SourceClass, SourceEventConfig

GLASS = [("glass", 3.2e-3, 2500.0, 72.0e9, 0.22)]


@pytest.mark.parametrize("k", [150.0, 300.0, 600.0])
def test_single_layer_safe_matches_exact_lamb(k):
    """SAFE on a single glass plate must reproduce the exact Rayleigh-Lamb roots (<1.5 %)
    across the AE band (k = 150-600 rad/m ~ 18-180 kHz for A0)."""
    safe = MultilayerSAFESolver(layers=GLASS, n_elem_per_layer=12)
    lamb = LambDispersionSolver(thickness_m=3.2e-3)
    f_s0, cp_s0, f_a0, cp_a0 = safe.compute_dispersion(np.array([k]))
    assert abs(cp_s0[0] - lamb.find_phase_velocity_s0(f_s0[0])) / cp_s0[0] < 0.015
    assert abs(cp_a0[0] - lamb.find_phase_velocity_a0(f_a0[0])) / cp_a0[0] < 0.015


def test_safe_a0_low_frequency_matches_thin_plate_theory():
    """At very low k the A0 branch must match Kirchhoff thin-plate theory c = k sqrt(D/rho h)."""
    E, nu, rho, h, k = 72.0e9, 0.22, 2500.0, 3.2e-3, 50.0
    D = E * h**3 / (12 * (1 - nu**2))
    safe = MultilayerSAFESolver(layers=GLASS, n_elem_per_layer=12)
    _, _, _, cp_a0 = safe.compute_dispersion(np.array([k]))
    assert cp_a0[0] == pytest.approx(k * np.sqrt(D / (rho * h)), rel=0.01)


def test_safe_element_matrix_is_hermitian():
    safe = MultilayerSAFESolver(n_elem_per_layer=2)
    K, _ = safe.assemble_matrices(200.0)
    assert np.allclose(K, K.conj().T)


def test_safe_s0_never_exceeds_plate_speed():
    safe = MultilayerSAFESolver(layers=GLASS, n_elem_per_layer=8)
    _, cp_s0, _, _ = safe.compute_dispersion(np.array([20.0, 100.0, 400.0]))
    assert np.all(cp_s0 <= safe.plate_velocity_limit() * 1.001)


def test_laminate_s0_close_to_bare_glass():
    """Thin soft layers only mass-load the glass: S0 within a few % of bare glass (not -40 %)."""
    lam = laminate_for_module(ModuleTwinConfig())
    v = lam.s0_velocity(25.0)
    assert 5200.0 < v < 5600.0


def test_wlf_shift_stiffens_more_than_arrhenius_in_cold():
    hybrid = ViscoelasticEVA(shift_model="hybrid")
    arr = ViscoelasticEVA(shift_model="arrhenius")
    assert hybrid.shift_factor(-20.0) > arr.shift_factor(-20.0)
    # identical above the switch temperature
    assert hybrid.shift_factor(40.0) == pytest.approx(arr.shift_factor(40.0))


@pytest.fixture(scope="module")
def sim():
    return Tier0Simulator()


def test_crack_couples_more_s0_than_hail(sim):
    crack = SourceEventConfig(x_mm=520, y_mm=880)
    hail = SourceEventConfig(source_class=SourceClass.NOISE, noise_class=NoiseClass.HAIL, x_mm=520, y_mm=880)
    assert sim.mode_coupling_ratio(crack) > 20.0 * sim.mode_coupling_ratio(hail)


def test_first_arrival_matches_s0_speed(sim):
    onset = 400e-6
    w = sim.simulate(SourceEventConfig(x_mm=300, y_mm=500), 25.0, np.random.default_rng(1), onset_s=onset)
    v = sim.s0_velocity(25.0)
    for ch, (sx, sy) in enumerate(sim.sensor_config.positions_mm):
        d = np.hypot(300 - sx, 500 - sy) * 1e-3
        # 0.3 % of peak: the S0 head wave is much weaker than the later A0 packet (the
        # causality mute guarantees nothing earlier than the physical arrival)
        first = np.argmax(np.abs(w[ch]) > 0.003 * np.max(np.abs(w[ch])))
        assert abs(first * 1e-6 - (onset + d / v)) < 15e-6


def test_noise_only_pretrigger_segment(sim):
    w = sim.simulate(SourceEventConfig(x_mm=520, y_mm=880), 25.0, np.random.default_rng(2), onset_s=400e-6)
    assert np.max(np.abs(w[:, :380])) < 0.01 * np.max(np.abs(w))


def test_edge_reflections_add_coda():
    cfg = ModuleTwinConfig()
    with_refl = Tier0Simulator(cfg, edge_reflection_coeff=0.6)
    no_refl = Tier0Simulator(cfg, edge_reflection_coeff=0.0)
    src = SourceEventConfig(x_mm=200, y_mm=300)
    a = with_refl.simulate(src, 25.0, np.random.default_rng(3))
    b = no_refl.simulate(src, 25.0, np.random.default_rng(3))
    # Sensor 0 at (60, 60): direct path 278 mm; first image (left edge) path 354 mm.
    v = with_refl.s0_velocity(25.0)
    t_image = int(0.354 / v * 1e6)
    diff = a[0] - b[0]
    # causal: identical before the first echo, up to the documented ~-40 dB numerical floor of
    # image paths (the causality mute is applied to the direct path only)
    assert np.max(np.abs(diff[: t_image - 8])) < 0.03 * np.max(np.abs(a[0]))
    assert np.sum(diff[t_image - 8:] ** 2) > 0.05 * np.sum(b[0] ** 2)          # echoes present


def test_emi_is_simultaneous_on_all_channels(sim):
    emi = SourceEventConfig(source_class=SourceClass.NOISE, noise_class=NoiseClass.EMI_BURST, x_mm=100, y_mm=100)
    w = sim.simulate(emi, 25.0, np.random.default_rng(4))
    for ch in range(1, w.shape[0]):
        np.testing.assert_allclose(w[ch], w[0])


def test_sensor_positions_scale_with_module():
    cfg = SensorArrayConfig.for_module(1134.0, 2278.0)
    xs = [p[0] for p in cfg.positions_mm]
    ys = [p[1] for p in cfg.positions_mm]
    assert max(xs) == pytest.approx(1074.0)
    assert max(ys) == pytest.approx(2218.0)


def test_tier1_heterogeneous_reciprocity():
    """Reciprocity must hold with a real velocity inclusion (the old test never applied it)."""
    solver = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.01, dt_s=1e-6, pml_thickness_cells=6)
    c = np.full((solver.nx, solver.ny), 5300.0)
    c[22:30, 10:20] = 3800.0
    solver.set_c_field(c)
    t = np.arange(220) * 1e-6
    f0 = 120e3
    tau = t - 2.0 / f0
    src = (1 - 2 * (np.pi * f0 * tau) ** 2) * np.exp(-((np.pi * f0 * tau) ** 2))
    a, b = (0.12, 0.15), (0.31, 0.30)
    ab = solver.run_simulation(*a, src, [b])[0][0]
    ba = solver.run_simulation(*b, src, [a])[0][0]
    assert np.linalg.norm(ab - ba) / np.linalg.norm(ab) < 1e-6
    # and the inclusion must actually change the field
    homog = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.01, dt_s=1e-6, pml_thickness_cells=6)
    ab0 = homog.run_simulation(*a, src, [b])[0][0]
    assert np.linalg.norm(ab - ab0) / np.linalg.norm(ab0) > 0.01
