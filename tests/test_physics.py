import numpy as np
from acoustisolar_physics.materials import GLASS_MATERIAL
from acoustisolar_physics.lamb import LambDispersionSolver
from acoustisolar_physics.safe import MultilayerSAFESolver
from acoustisolar_physics.viscoelastic_eva import ViscoelasticEVA
from acoustisolar_physics.tier0 import Tier0Simulator
from acoustisolar_schemas.source import SourceEventConfig, SourceClass, CrackType

def test_material_acoustic_speeds():
    mat = GLASS_MATERIAL
    assert mat.density == 2500.0
    # Bulk longitudinal speed ~ 5690 m/s, shear ~ 3436 m/s
    assert 5500 < mat.c_longitudinal < 5900
    assert 3300 < mat.c_transverse < 3600
    # S0 low frequency limit ~ 5300-5500 m/s
    assert 5100 < mat.c_plate_s0_limit < 5600

def test_lamb_dispersion_limits():
    solver = LambDispersionSolver(thickness_m=3.2e-3)
    # At low frequency (e.g. 20 kHz), cp_S0 should be very close to theoretical c_plate limit
    cp_s0_low = solver.find_phase_velocity_s0(20.0e3)
    c_lim = GLASS_MATERIAL.c_plate_s0_limit
    assert abs(cp_s0_low - c_lim) / c_lim < 0.05
    
    # A0 flexural mode at low frequency should be significantly slower than S0
    cp_a0_low = solver.find_phase_velocity_a0(50.0e3)
    assert cp_a0_low < cp_s0_low
    assert cp_a0_low > 200.0

def test_viscoelastic_eva_temperature_dependence():
    eva = ViscoelasticEVA()
    # Loss factor at room temperature vs cold (-10 C) vs hot (65 C)
    g_p_25, g_pp_25 = eva.complex_shear_modulus(100.0e3, temp_c=25.0)
    g_p_cold, g_pp_cold = eva.complex_shear_modulus(100.0e3, temp_c=-10.0)
    g_p_hot, g_pp_hot = eva.complex_shear_modulus(100.0e3, temp_c=65.0)
    
    # Colder temperature -> higher stiffness (higher storage modulus)
    assert g_p_cold > g_p_25 > g_p_hot
    
    # Attenuation should be positive
    alpha = eva.attenuation_factor(np.array([100.0e3, 200.0e3]), temp_c=25.0)
    assert np.all(alpha >= 0.0)

def test_safe_solver_modes():
    safe = MultilayerSAFESolver(n_elem_per_layer=2)
    # Solve at k = 100 rad/m
    freqs = safe.solve_modes_at_wavenumber(100.0, n_modes=2)
    assert len(freqs) == 2
    assert freqs[0] > 0
    assert freqs[1] > freqs[0]

def test_tier0_simulation_and_reciprocity():
    sim = Tier0Simulator()
    src = SourceEventConfig(
        source_class=SourceClass.CRACK_INITIATION,
        crack_type=CrackType.DIAGONAL,
        x_mm=500.0,
        y_mm=800.0,
    )
    waves = sim.simulate(src)
    assert waves.shape == (4, 2560)
    # Waveform should be non-zero and finite
    assert not np.isnan(waves).any()
    assert np.max(np.abs(waves)) > 0.0
    
    # Test causality: arrival at nearer sensor should be earlier or equal to farther sensor
    dists = [
        np.hypot(500.0 - pos[0], 800.0 - pos[1])
        for pos in sim.sensor_config.positions_mm
    ]
    min_dist_idx = np.argmin(dists)
    max_dist_idx = np.argmax(dists)
    
    # Time of maximum energy in nearer sensor should be <= farther sensor
    t_peak_near = np.argmax(np.abs(waves[min_dist_idx]))
    t_peak_far = np.argmax(np.abs(waves[max_dist_idx]))
    assert t_peak_near <= t_peak_far + 20
