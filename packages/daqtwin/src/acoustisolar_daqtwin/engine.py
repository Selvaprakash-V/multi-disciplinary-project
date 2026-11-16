import numpy as np
from typing import Optional, Dict, Any
from acoustisolar_schemas.daq import DAQConfig
from acoustisolar_schemas.sensor import SensorArrayConfig
from .pzt_sensor import PZTSensorModel
from .analog_chain import AnalogFrontEnd
from .adc import ADCTwin
from .trigger import TriggerEngine

class DAQDigitalTwin:
    """
    Complete end-to-end hardware twin:
    PZT Sensors -> Preamp & Filters -> ADC Sampling -> Trigger Processing.
    """

    def __init__(
        self,
        daq_config: Optional[DAQConfig] = None,
        sensor_config: Optional[SensorArrayConfig] = None,
        seed: int = 42
    ):
        self.daq_config = daq_config or DAQConfig()
        self.sensor_config = sensor_config or SensorArrayConfig()
        self.seed = seed
        
        self.pzt_sensors = [
            PZTSensorModel(self.sensor_config.disc, seed=seed + i)
            for i in range(self.sensor_config.count)
        ]
        self.analog_fe = AnalogFrontEnd(self.daq_config.preamp, seed=seed + 100)
        self.adc = ADCTwin(self.daq_config.adc, seed=seed + 200)
        self.trigger = TriggerEngine(self.daq_config.trigger, sample_rate_hz=self.daq_config.adc.rate_hz)

    def process_mechanical_waves(
        self,
        surface_velocities: np.ndarray,
        channel_skew_ns: Optional[np.ndarray] = None,
        interference_v: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Takes raw simulated acoustic velocities (m/s) at sensor locations and simulates the DAQ chain.
        ``interference_v`` is electrical pickup injected at the preamp input (bypasses the sensors).
        Returns:
            dict containing raw_counts, calibrated_volts, triggered, trigger_ch, trigger_idx, windowed_counts
        """
        n_channels = surface_velocities.shape[0]
        fs = self.daq_config.adc.rate_hz

        # 1. PZT Transduction
        sensor_volts = np.zeros_like(surface_velocities)
        for ch in range(n_channels):
            sensor_volts[ch] = self.pzt_sensors[ch].apply_transduction(surface_velocities[ch], fs)

        # 2. Analog Preamp & (causal) Filtering
        amplified_volts = self.analog_fe.process(sensor_volts, fs, interference_v=interference_v)
        
        # 3. ADC Sampling and Digitization
        raw_counts = self.adc.digitize(amplified_volts, channel_skew_ns=channel_skew_ns)
        calibrated_volts = self.adc.counts_to_volts(raw_counts)
        
        # 4. Trigger Evaluation
        triggered, trig_ch, trig_idx = self.trigger.evaluate_trigger(calibrated_volts)
        
        windowed_counts = None
        windowed_volts = None
        if triggered:
            windowed_counts = self.trigger.extract_event_window(raw_counts, trig_idx)
            windowed_volts = self.trigger.extract_event_window(calibrated_volts, trig_idx)
            
        return {
            "sensor_volts": sensor_volts,
            "amplified_volts": amplified_volts,
            "raw_counts": raw_counts,
            "calibrated_volts": calibrated_volts,
            "triggered": triggered,
            "trigger_channel": trig_ch,
            "trigger_sample_idx": trig_idx,
            "windowed_counts": windowed_counts,
            "windowed_volts": windowed_volts,
        }

