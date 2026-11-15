import pytest
import numpy as np
from acoustisolar_physics.tier1 import Tier1WaveSolver

def test_tier1_cfl_condition():
    # dx = 0.02 m, dt = 2e-6 s, c = 5300 m/s -> Courant = 5300 * 2e-6 / 0.02 = 0.53 <= 0.707 (stable)
    solver = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.02, dt_s=2e-6, c_speed_m_s=5300.0)
    assert solver.courant_number <= 0.707
    
    # Violating CFL should raise AssertionError
    with pytest.raises(AssertionError):
        Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.01, dt_s=3e-6, c_speed_m_s=5300.0) # Courant = 1.59 > 0.707

def test_tier1_wave_propagation_and_energy():
    solver = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.02, dt_s=2e-6, c_speed_m_s=5300.0)
    # Source at center (0.2, 0.2), sensor at (0.1, 0.1) and (0.3, 0.3)
    n_steps = 150
    t = np.arange(n_steps) * 2e-6
    source = np.exp(-((t - 15e-6) / 5e-6)**2) * np.sin(2 * np.pi * 100e3 * t)
    
    sensors = [(0.1, 0.1), (0.3, 0.3)]
    waves, max_energy = solver.run_simulation(
        source_x_m=0.2,
        source_y_m=0.2,
        source_signal=source,
        sensor_locations_m=sensors,
    )
    assert waves.shape == (2, n_steps)
    assert max_energy > 0.0
    # Symmetric placement should yield approximately equal peak amplitudes
    peak1 = np.max(np.abs(waves[0]))
    peak2 = np.max(np.abs(waves[1]))
    assert pytest.approx(peak1, rel=0.1) == peak2

def test_tier1_fidelity_gap():
    solver = Tier1WaveSolver(width_m=0.4, length_m=0.4, dx_m=0.02, dt_s=2e-6, c_speed_m_s=5300.0)
    n = 200
    w0 = np.random.normal(0, 0.1, (2, n))
    w1 = w0 + np.random.normal(0, 0.01, (2, n))
    gap = solver.compare_fidelity_gap(w0, w1)
    assert "arrival_time_error_pct" in gap
    assert "spectral_centroid_error_pct" in gap
    assert "envelope_correlation" in gap
    assert gap["envelope_correlation"] > 0.8

