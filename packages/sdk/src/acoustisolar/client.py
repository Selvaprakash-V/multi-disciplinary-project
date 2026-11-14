import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from acoustisolar_schemas.module import ModuleTwinConfig
from acoustisolar_schemas.sensor import SensorArrayConfig
from acoustisolar_schemas.daq import DAQConfig
from acoustisolar_schemas.source import SourceEventConfig, SourceClass, CrackType
from acoustisolar_physics.tier0 import Tier0Simulator
from acoustisolar_physics.tier1 import Tier1WaveSolver
from acoustisolar_daqtwin.engine import DAQDigitalTwin
from acoustisolar_dsp.filter import DigitalBandpassFilter
from acoustisolar_dsp.pickers import OnsetPicker
from acoustisolar_dsp.features import AEFeatureExtractor
from acoustisolar_dsp.tdoa import TDOALocalizer
from acoustisolar_ml.generator import DatasetGenerator
from acoustisolar_ml.baseline_model import AEBaselineGradientBooster
from acoustisolar_ml.evaluation import HeldOutEvaluator

class AcoustiSolar:
    """
    High-level Python SDK for the AcoustiSolar platform.
    Integrates multi-fidelity simulation, DAQ twin, signal processing, and ML.
    """

    def __init__(
        self,
        module_config: Optional[ModuleTwinConfig] = None,
        sensor_config: Optional[SensorArrayConfig] = None,
        daq_config: Optional[DAQConfig] = None,
        seed: int = 42,
    ):
        self.module = module_config or ModuleTwinConfig()
        self.sensors = sensor_config or SensorArrayConfig()
        self.daq = daq_config or DAQConfig()
        self.seed = seed
        
        self.t0_sim = Tier0Simulator(self.module, self.sensors, sample_rate_hz=self.daq.adc.rate_hz)
        self.daq_twin = DAQDigitalTwin(self.daq, self.sensors, seed=seed)
        self.extractor = AEFeatureExtractor(sample_rate_hz=self.daq.adc.rate_hz)
        self.localizer = TDOALocalizer(self.sensors.positions_mm, nominal_velocity_m_s=5300.0)

    def simulate_event(
        self,
        x_mm: float = 500.0,
        y_mm: float = 800.0,
        source_class: SourceClass = SourceClass.CRACK_INITIATION,
        crack_type: CrackType = CrackType.DIAGONAL,
        temperature_c: float = 25.0,
    ) -> Dict[str, Any]:
        """Simulate wave propagation and complete DAQ acquisition for a single event."""
        src = SourceEventConfig(
            source_class=source_class,
            crack_type=crack_type,
            x_mm=x_mm,
            y_mm=y_mm,
        )
        mech_waves = self.t0_sim.simulate(src, temperature_c=temperature_c)
        daq_out = self.daq_twin.process_mechanical_waves(mech_waves)
        
        # Localize if triggered
        loc_res = None
        if daq_out["triggered"]:
            cal_volts = daq_out["calibrated_volts"]
            dt_s = self.localizer.estimate_time_differences(cal_volts)
            loc_res = self.localizer.solve_hyperbolic(dt_s)
            
        return {
            "mechanical_waves": mech_waves,
            "calibrated_volts": daq_out["calibrated_volts"],
            "raw_counts": daq_out["raw_counts"],
            "triggered": daq_out["triggered"],
            "trigger_channel": daq_out["trigger_channel"],
            "localization": loc_res,
        }

    def generate_synthetic_dataset(
        self,
        n_events: int = 50,
        crack_ratio: float = 0.6,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Generate a labeled synthetic training dataset."""
        generator = DatasetGenerator(
            module_config=self.module,
            sensor_config=self.sensors,
            daq_config=self.daq,
            seed=self.seed,
        )
        return generator.generate_batch(n_events=n_events, crack_ratio=crack_ratio)

    def evaluate_held_out(self, n_train: int = 50, n_test: int = 25) -> Dict[str, Any]:
        """Run leakage-free generalization evaluation across held-out module configurations."""
        evaluator = HeldOutEvaluator(seed=self.seed)
        return evaluator.run_held_out_benchmark(n_train=n_train, n_test=n_test)
