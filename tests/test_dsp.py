import numpy as np
import pytest

from acoustisolar_dsp import AEFeatureExtractor, DeltaTMapLocalizer, DigitalBandpassFilter, OnsetPicker, TDOALocalizer

SENSORS = [(60.0, 60.0), (60.0, 1700.0), (980.0, 60.0), (980.0, 1700.0)]


def _burst(n, onset, rng=None, noise=0.02, f=0.15):
    rng = rng or np.random.default_rng(0)
    t_rel = np.maximum(np.arange(n) - onset, 0)
    sig = np.exp(-t_rel / 30.0) * np.sin(2 * np.pi * f * t_rel) * (np.arange(n) >= onset)
    return sig + rng.normal(0, noise, n)


def test_bandpass_filter():
    filt = DigitalBandpassFilter(low_cut_hz=50.0e3, high_cut_hz=300.0e3, sample_rate_hz=1_000_000.0)
    t = np.arange(1000) / 1e6
    out = {f: np.max(np.abs(filt.filter(np.sin(2 * np.pi * f * t)))) for f in (10e3, 150e3, 450e3)}
    assert out[150e3] > 3.0 * out[10e3]
    assert out[150e3] > 1.5 * out[450e3]


def test_vectorised_aic_matches_bruteforce():
    rng = np.random.default_rng(3)
    x = _burst(600, 300, rng, noise=0.05)
    w1, w2, m = 200, 400, 5
    brute = []
    for k in range(w1 + m, w2 - m):
        brute.append((k - w1) * np.log(np.var(x[w1:k]) + 1e-30) + (w2 - k) * np.log(np.var(x[k:w2]) + 1e-30))
    expected = w1 + m + int(np.argmin(brute))
    assert OnsetPicker.pick_aic(x, (w1, w2)) == expected
    assert abs(expected - 300) <= 5


def test_sta_lta_is_causal():
    x = _burst(1000, 400)
    pick = OnsetPicker.pick_sta_lta(x, n_sta=10, n_lta=80, threshold=3.0)
    assert 395 <= pick <= 420
    # changing the future must not change the pick
    y = x.copy()
    y[600:] = 0.0
    assert OnsetPicker.pick_sta_lta(y, n_sta=10, n_lta=80, threshold=3.0) == pick


def test_ae_feature_extractor_extended_fields():
    ex = AEFeatureExtractor(sample_rate_hz=1e6, threshold_volts=0.02)
    t = np.arange(1000) / 1e6
    sig = 0.1 * np.exp(-t / 50e-6) * np.sin(2 * np.pi * 200e3 * t)
    f = ex.extract_features(sig)
    assert pytest.approx(f.peak_amp_mv, rel=0.1) == 100.0
    assert 160.0 < f.centroid_khz < 240.0
    assert abs(sum(f.partial_powers) - 1.0) < 0.05
    assert f.partial_powers[2] > 0.5   # 150-250 kHz band dominates
    assert f.ra_us_per_mv >= 0.0 and f.avg_freq_khz > 0.0


def test_event_features_arrival_spread_zero_for_simultaneous_signal():
    ex = AEFeatureExtractor()
    common = _burst(2560, 500, noise=0.0) * 0.2
    w = np.stack([common + np.random.default_rng(i).normal(0, 1e-4, 2560) for i in range(4)])
    ev = ex.extract_event_features(w)
    assert ev["event"]["arrival_spread_us"] <= 2.0


def test_tdoa_legacy_solver_exact_geometry():
    loc = TDOALocalizer(SENSORS, nominal_velocity_m_s=5300.0)
    src = (500.0, 800.0)
    d = [np.hypot(src[0] - s[0], src[1] - s[1]) for s in SENSORS]
    dt = np.array([(di - d[0]) / 5.3e6 for di in d])
    res = loc.solve_hyperbolic(dt, initial_guess_mm=(450.0, 750.0))
    assert res["converged"] and np.hypot(res["x_mm"] - 500, res["y_mm"] - 800) < 5.0


def test_robust_solver_rejects_single_outlier():
    loc = TDOALocalizer(SENSORS + [(520.0, 60.0)], nominal_velocity_m_s=5400.0)
    src = np.array([300.0, 1200.0])
    S = np.array(loc.sensors_mm)
    t = 1e-4 + np.hypot(*(S - src).T) / 5.4e6
    t[1] += 120e-6  # A0-like late pick on one channel
    res = loc.solve_arrivals(t, bounds_mm=(0, 1040, 0, 1760))
    assert np.hypot(res["x_mm"] - src[0], res["y_mm"] - src[1]) < 40.0


def test_consistency_violation_detected():
    loc = TDOALocalizer(SENSORS)
    t = np.array([0.0, 1e-3, 0.0, 0.0])  # 1 ms difference across 1.64 m is impossible at 5.4 km/s
    assert loc.consistency_violations(t, 5400.0)[0, 1]


def test_gated_locate_on_simulated_crack():
    from acoustisolar_daqtwin.engine import DAQDigitalTwin
    from acoustisolar_physics.tier0 import Tier0Simulator
    from acoustisolar_schemas.source import SourceEventConfig

    sim = Tier0Simulator()
    daq = DAQDigitalTwin(sensor_config=sim.sensor_config)
    loc = TDOALocalizer(sim.sensor_config.positions_mm)
    errs = []
    for i, (x, y) in enumerate([(300, 500), (700, 1200), (520, 880)]):
        src = SourceEventConfig(x_mm=x, y_mm=y, energy_arbitrary=3.0)
        mech = sim.simulate(src, 25.0, np.random.default_rng(i), onset_s=400e-6)
        volts = daq.process_mechanical_waves(mech)["calibrated_volts"]
        r = loc.locate(volts, v_eff_m_s=sim.s0_velocity(25.0), bounds_mm=(0, 1040, 0, 1760))
        errs.append(np.hypot(r["x_mm"] - x, r["y_mm"] - y))
    assert np.median(errs) < 30.0


def test_delta_t_map_localizer():
    S = np.array(SENSORS)

    def arrivals(x, y, v=5400e3):
        return np.hypot(S[:, 0] - x, S[:, 1] - y) / v

    dtm = DeltaTMapLocalizer(np.linspace(0, 1040, 14), np.linspace(0, 1760, 23)).calibrate(arrivals)
    res = dtm.locate(arrivals(410.0, 930.0) + 1e-4)
    assert np.hypot(res["x_mm"] - 410, res["y_mm"] - 930) < 25.0
