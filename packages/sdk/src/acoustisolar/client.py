from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from acoustisolar_daqtwin.engine import DAQDigitalTwin
from acoustisolar_dsp.features import AEFeatureExtractor
from acoustisolar_dsp.tdoa import TDOALocalizer
from acoustisolar_ml.evaluation import HeldOutEvaluator
from acoustisolar_ml.generator import DatasetGenerator
from acoustisolar_physics.tier0 import Tier0Simulator
from acoustisolar_schemas.daq import DAQConfig
from acoustisolar_schemas.module import ModuleTwinConfig
from acoustisolar_schemas.sensor import SensorArrayConfig
from acoustisolar_schemas.source import CrackType, NoiseClass, SourceClass, SourceEventConfig


class AcoustiSolar:
    """High-level SDK: Tier-0 simulation, DAQ twin, localisation, features and ML."""

    def __init__(
        self,
        module_config: Optional[ModuleTwinConfig] = None,
        sensor_config: Optional[SensorArrayConfig] = None,
        daq_config: Optional[DAQConfig] = None,
        seed: int = 42,
    ):
        self.module = module_config or ModuleTwinConfig()
        self.sensors = sensor_config or SensorArrayConfig.for_module(self.module.width_mm, self.module.length_mm)
        self.daq = daq_config or DAQConfig()
        self.seed = seed
        n = self.daq.trigger.total_samples
        self.t0_sim = Tier0Simulator(self.module, self.sensors, sample_rate_hz=self.daq.adc.rate_hz, n_samples=n)
        self.daq_twin = DAQDigitalTwin(self.daq, self.sensors, seed=seed)
        self.extractor = AEFeatureExtractor(sample_rate_hz=self.daq.adc.rate_hz)
        self.localizer = TDOALocalizer(self.sensors.positions_mm, sample_rate_hz=self.daq.adc.rate_hz)

    def simulate_event(
        self,
        x_mm: float = 500.0,
        y_mm: float = 800.0,
        source_class: SourceClass = SourceClass.CRACK_INITIATION,
        crack_type: CrackType = CrackType.DIAGONAL,
        temperature_c: float = 25.0,
        noise_class: NoiseClass = NoiseClass.NONE,
        energy: float = 1.0,
        onset_us: float = 400.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Simulate propagation + DAQ acquisition for one event and localise it."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        src = SourceEventConfig(source_class=source_class, crack_type=crack_type, noise_class=noise_class,
                                x_mm=x_mm, y_mm=y_mm, energy_arbitrary=energy)
        mech = self.t0_sim.simulate(src, temperature_c=temperature_c, rng=rng, onset_s=onset_us * 1e-6)
        daq_out = self.daq_twin.process_mechanical_waves(mech)
        loc = None
        if daq_out["triggered"]:
            loc = self.localizer.locate(
                daq_out["calibrated_volts"],
                v_eff_m_s=self.t0_sim.s0_velocity(temperature_c),
                bounds_mm=(0.0, self.module.width_mm, 0.0, self.module.length_mm),
            )
        return {
            "mechanical_waves": mech,
            "calibrated_volts": daq_out["calibrated_volts"],
            "raw_counts": daq_out["raw_counts"],
            "triggered": daq_out["triggered"],
            "trigger_channel": daq_out["trigger_channel"],
            "localization": loc,
        }

    def generate_synthetic_dataset(self, n_events: int = 50, crack_ratio: float = 0.6) -> Tuple[np.ndarray, pd.DataFrame]:
        """Labelled synthetic dataset (multi-class ``event_class`` + binary ``is_crack``)."""
        generator = DatasetGenerator(module_config=self.module, sensor_config=self.sensors,
                                     daq_config=self.daq, seed=self.seed)
        return generator.generate_batch(n_events=n_events, crack_ratio=crack_ratio)

    def evaluate_held_out(self, n_train: int = 50, n_test: int = 25) -> Dict[str, Any]:
        """Generalisation to an unseen module design (train and test modules differ physically)."""
        return HeldOutEvaluator(seed=self.seed).run_held_out_benchmark(n_train=n_train, n_test=n_test)
