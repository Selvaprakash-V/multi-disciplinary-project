import numpy as np
import pytest

from acoustisolar_physics.tier1 import Tier1WaveSolver


def _pulse(n_steps, dt, f0=100e3, t0=15e-6, width=5e-6):
    t = np.arange(n_steps) * dt
    return np.exp(-(((t - t0) / width) ** 2)) * np.sin(2 * np.pi * f0 * t)


def test_tier1_cfl_condition():
    solver = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.02, dt_s=2e-6, c_speed_m_s=5300.0)
    assert solver.courant_number <= 0.707
    with pytest.raises(AssertionError):
        Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.01, dt_s=3e-6, c_speed_m_s=5300.0)


def test_tier1_wave_propagation_and_energy():
    # 3-cell sponge (60 mm) keeps the sensors outside the absorber
    solver = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.02, dt_s=2e-6, c_speed_m_s=5300.0,
                             pml_thickness_cells=3)
    source = _pulse(150, 2e-6)
    waves, max_energy = solver.run_simulation(0.2, 0.2, source, [(0.1, 0.1), (0.3, 0.3)])
    assert waves.shape == (2, 150)
    assert max_energy > 0.0
    assert pytest.approx(np.max(np.abs(waves[0])), rel=0.1) == np.max(np.abs(waves[1]))


def test_tier1_rejects_sensor_inside_sponge():
    """Regression: the KPI benchmark once placed sensors inside the absorbing layer."""
    solver = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.02, dt_s=2e-6, pml_thickness_cells=10)
    with pytest.raises(ValueError, match="sponge"):
        solver.run_simulation(0.2, 0.2, _pulse(50, 2e-6), [(0.1, 0.1)])


def test_tier1_reflecting_boundary_produces_echoes():
    kw = dict(dx_m=0.005, dt_s=0.5e-6, c_speed_m_s=5300.0, boundary="reflecting")
    src = _pulse(300, 0.5e-6, f0=150e3, t0=10e-6, width=3e-6)
    refl = Tier1WaveSolver(width_m=0.3, length_m=0.3, **kw).run_simulation(0.15, 0.15, src, [(0.24, 0.15)])[0][0]
    # echo-free reference: large domain with the same relative source/sensor geometry
    ref = Tier1WaveSolver(width_m=1.2, length_m=1.2, **kw).run_simulation(0.6, 0.6, src, [(0.69, 0.6)])[0][0]
    diff = np.abs(refl - ref)
    peak = np.max(np.abs(refl))
    # right-edge echo path 0.15 + 0.06 m -> ~40 us after emission (+10 us pulse delay)
    assert np.max(diff[: int(38e-6 / 0.5e-6)]) < 0.01 * peak      # identical before the echo
    assert np.max(diff[int(40e-6 / 0.5e-6):]) > 0.05 * peak       # echo present afterwards


def test_tier1_fidelity_gap_identical_fields_agree():
    solver = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.02, dt_s=2e-6, c_speed_m_s=5300.0)
    w = np.zeros((2, 200))
    w[:, 50:60] = np.hanning(10)
    gap = solver.compare_fidelity_gap(w, w)
    assert gap["arrival_time_error_pct"] == 0.0
    assert gap["envelope_correlation"] > 0.999
