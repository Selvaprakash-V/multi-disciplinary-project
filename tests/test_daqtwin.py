import pytest
import numpy as np
from acoustisolar_daqtwin import (
    PZTSensorModel,
    AnalogFrontEnd,
    ADCTwin,
    TriggerEngine,
    DAQDigitalTwin,
)
from acoustisolar_schemas.sensor import PZTDiscConfig
from acoustisolar_schemas.daq import PreampConfig, ADCConfig, TriggerConfig

def test_pzt_sensor_transduction():
    pzt = PZTSensorModel()
    n = 1000
    fs = 1_000_000.0
    t = np.arange(n) / fs
    # 200 kHz input mechanical velocity
    v_in = 1e-4 * np.sin(2.0 * np.pi * 200.0e3 * t)
    v_out = pzt.apply_transduction(v_in, fs)
    assert len(v_out) == n
    assert np.max(np.abs(v_out)) > 0.0

def test_analog_front_end():
    preamp_cfg = PreampConfig(gain_db=40.0, band_khz=(50.0, 400.0))
    afe = AnalogFrontEnd(preamp_cfg)
    fs = 1_000_000.0
    n = 2000
    t = np.arange(n) / fs
    # 150 kHz tone inside bandpass
    v_in = 1e-3 * np.sin(2.0 * np.pi * 150.0e3 * t)
    v_out = afe.process(v_in, fs)
    # 40 dB gain is 100x voltage amplification
    peak_in = np.max(np.abs(v_in))
    peak_out = np.max(np.abs(v_out))
    assert 70.0 < (peak_out / peak_in) < 130.0

def test_adc_quantization_and_clipping():
    adc_cfg = ADCConfig(bits=12, full_scale_volts=3.3)
    adc = ADCTwin(adc_cfg)
    v_in = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) # Exceeds +/- 1.65V
    counts = adc.digitize(v_in)
    assert counts.dtype == np.int16
    assert np.all(counts >= 0)
    assert np.all(counts <= 4095)
    # Check clipping at extremes
    assert counts[0] == 0
    assert counts[-1] == 4095

def test_trigger_engine_windowing():
    trig_cfg = TriggerConfig(threshold_mv=50.0, pre_samples=100, post_samples=200)
    engine = TriggerEngine(trig_cfg, sample_rate_hz=1_000_000.0)
    
    # 4 channels, impulse at sample 300 on channel 1
    waveforms = np.zeros((4, 1000))
    waveforms[1, 300] = 0.1 # 100 mV (exceeds 50 mV threshold)
    
    trig, ch, idx = engine.evaluate_trigger(waveforms)
    assert trig is True
    assert ch == 1
    assert idx == 300
    
    window = engine.extract_event_window(waveforms, idx)
    assert window.shape == (4, 300)
    # Trigger point should be at index 100 (pre_samples)
    assert window[1, 100] == 0.1

def test_full_daq_digital_twin():
    twin = DAQDigitalTwin()
    fs = twin.daq_config.adc.rate_hz
    n = 2560
    t = np.arange(n) / fs
    
    # Simulate surface acoustic waves on 4 sensors
    waves = np.zeros((4, n))
    for i in range(4):
        delay = i * 20e-6
        t_rel = np.maximum(t - delay, 0)
        waves[i] = 1e-4 * np.exp(-t_rel / 30e-6) * np.sin(2*np.pi*180e3*t_rel) * (t >= delay)
        
    out = twin.process_mechanical_waves(waves)
    assert out["raw_counts"].shape == (4, n)
    assert out["triggered"] is True
    assert out["windowed_counts"] is not None
