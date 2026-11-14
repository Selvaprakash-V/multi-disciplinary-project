import pytest
import numpy as np
from acoustisolar_dsp import (
    DigitalBandpassFilter,
    OnsetPicker,
    AEFeatureExtractor,
    TDOALocalizer,
)

def test_bandpass_filter():
    filt = DigitalBandpassFilter(low_cut_hz=50.0e3, high_cut_hz=300.0e3, sample_rate_hz=1_000_000.0)
    fs = 1_000_000.0
    t = np.arange(1000) / fs
    # Low freq 10 kHz (out), In-band 150 kHz, High freq 450 kHz (out)
    sig_low = np.sin(2.0 * np.pi * 10.0e3 * t)
    sig_mid = np.sin(2.0 * np.pi * 150.0e3 * t)
    sig_high = np.sin(2.0 * np.pi * 450.0e3 * t)
    
    out_low = filt.filter(sig_low)
    out_mid = filt.filter(sig_mid)
    out_high = filt.filter(sig_high)
    
    # In-band should have higher amplitude than out-of-band attenuated signals
    assert np.max(np.abs(out_mid)) > 3.0 * np.max(np.abs(out_low))
    assert np.max(np.abs(out_mid)) > 1.5 * np.max(np.abs(out_high))

def test_aic_and_sta_lta_pickers():
    n = 1000
    noise = np.random.normal(0, 0.05, n)
    # Add arrival starting sharply at sample 400
    arrival_idx = 400
    t_rel = np.maximum(np.arange(n) - arrival_idx, 0)
    burst = 1.0 * np.exp(-t_rel / 30.0) * np.sin(2 * np.pi * 0.1 * t_rel) * (np.arange(n) >= arrival_idx)
    sig = noise + burst
    
    pick_aic = OnsetPicker.pick_aic(sig, search_window=(300, 500))
    # AIC pick should locate onset within +/- 5 samples
    assert abs(pick_aic - arrival_idx) <= 5
    
    pick_sta = OnsetPicker.pick_sta_lta(sig, n_sta=10, n_lta=80, threshold=3.0)
    assert abs(pick_sta - arrival_idx) <= 20

def test_ae_feature_extractor():
    extractor = AEFeatureExtractor(sample_rate_hz=1_000_000.0, threshold_volts=0.02)
    n = 1000
    t = np.arange(n) / 1_000_000.0
    sig = 0.100 * np.exp(-t / 50e-6) * np.sin(2 * np.pi * 200e3 * t)
    
    feats = extractor.extract_features(sig)
    assert pytest.approx(feats.peak_amp_mv, rel=0.1) == 100.0
    assert feats.counts > 0
    assert feats.energy_au > 0.0
    # Centroid frequency should be near 200 kHz
    assert 160.0 < feats.centroid_khz < 240.0

def test_tdoa_localization():
    sensors = [(60.0, 60.0), (60.0, 1640.0), (980.0, 60.0), (980.0, 1640.0)]
    localizer = TDOALocalizer(sensors, nominal_velocity_m_s=5300.0)
    
    # Ground truth source at (500, 800)
    src_true = (500.0, 800.0)
    v_mm_s = 5300.0 * 1000.0
    dists = [np.hypot(src_true[0] - s[0], src_true[1] - s[1]) for s in sensors]
    dt = np.array([(d - dists[0]) / v_mm_s for d in dists])
    
    res = localizer.solve_hyperbolic(dt, initial_guess_mm=(450.0, 750.0))
    assert res["converged"] is True
    # Error should be sub-millimeter for ideal geometric TDOA
    err = np.hypot(res["x_mm"] - src_true[0], res["y_mm"] - src_true[1])
    assert err < 5.0 # within 5 mm
