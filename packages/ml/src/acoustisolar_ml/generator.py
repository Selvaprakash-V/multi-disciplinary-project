import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from acoustisolar_schemas.module import ModuleTwinConfig
from acoustisolar_schemas.sensor import SensorArrayConfig
from acoustisolar_schemas.daq import DAQConfig
from acoustisolar_schemas.source import SourceEventConfig, SourceClass, CrackType, NoiseClass
from acoustisolar_schemas.event import EventRecord, TriggerFeatures, GroundTruthLabel, ProvenanceRecord
from acoustisolar_schemas.hashing import compute_config_hash
from acoustisolar_physics.tier0 import Tier0Simulator
from acoustisolar_daqtwin.engine import DAQDigitalTwin
from acoustisolar_dsp.features import AEFeatureExtractor

class DatasetGenerator:
    """
    Synthetic dataset generation engine.
    Produces bit-reproducible labeled acoustic emission datasets with domain randomization,
    waveforms, trigger features, and Parquet/DataFrame index.
    """

    def __init__(
        self,
        module_config: Optional[ModuleTwinConfig] = None,
        sensor_config: Optional[SensorArrayConfig] = None,
        daq_config: Optional[DAQConfig] = None,
        seed: int = 42,
    ):
        self.module_config = module_config or ModuleTwinConfig()
        self.sensor_config = sensor_config or SensorArrayConfig()
        self.daq_config = daq_config or DAQConfig()
        self.seed = seed
        self.config_hash = compute_config_hash(self.module_config)
        self.feature_extractor = AEFeatureExtractor(sample_rate_hz=self.daq_config.adc.rate_hz)
        
        # Instantiate simulation and DAQ engines once to avoid re-solving dispersion curves per event
        self.sim = Tier0Simulator(
            module_config=self.module_config,
            sensor_config=self.sensor_config,
            sample_rate_hz=self.daq_config.adc.rate_hz,
            n_samples=self.daq_config.trigger.total_samples,
        )
        self.daq_twin = DAQDigitalTwin(
            daq_config=self.daq_config,
            sensor_config=self.sensor_config,
            seed=self.seed,
        )

    def generate_event(
        self,
        event_id: int,
        source_class: SourceClass,
        crack_type: CrackType = CrackType.NONE,
        noise_class: NoiseClass = NoiseClass.NONE,
        temp_c: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a single simulated and DAQ-digitized event.
        Returns:
            (waveforms: np.ndarray shape (4, total_samples), metadata_dict)
        """
        if rng is None:
            rng = np.random.default_rng(self.seed + event_id)

        temperature = temp_c if temp_c is not None else float(rng.uniform(-10.0, 65.0))
        
        # Randomize source location inside module bounds
        width = self.module_config.width_mm
        length = self.module_config.length_mm
        margin = 80.0
        x_mm = float(rng.uniform(margin, width - margin))
        y_mm = float(rng.uniform(margin, length - margin))
        
        # Map (x, y) to cell ID
        cols = self.module_config.layout.cols
        rows = self.module_config.layout.rows
        cell_col = int(min(max(0, (x_mm / width) * cols), cols - 1))
        cell_row = int(min(max(0, (y_mm / length) * rows), rows - 1))
        cell_id = f"C_{cell_row}_{cell_col}"
        
        energy = float(rng.uniform(0.5, 3.0))
        rise_time = float(rng.uniform(0.8, 3.5))
        
        src_cfg = SourceEventConfig(
            source_class=source_class,
            crack_type=crack_type,
            noise_class=noise_class,
            x_mm=x_mm,
            y_mm=y_mm,
            cell_id=cell_id,
            energy_arbitrary=energy,
            rise_time_us=rise_time,
        )

        # 1. Physics simulation (Tier-0)
        mech_waves = self.sim.simulate(src_cfg, temperature_c=temperature, rng=rng)

        # 2. DAQ Twin processing
        channel_skew = np.array([float(rng.uniform(-10.0, 10.0)) for _ in range(self.sensor_config.count)])
        daq_out = self.daq_twin.process_mechanical_waves(mech_waves, channel_skew_ns=channel_skew)

        calibrated_volts = daq_out["calibrated_volts"]
        
        # 3. DSP feature extraction
        feats = self.feature_extractor.extract_features(calibrated_volts[0])
        feat_vector = self.feature_extractor.extract_channel_features(calibrated_volts)

        meta = {
            "event_id": f"EVT-{self.seed}-{event_id:06d}",
            "source_class": source_class.value,
            "crack_type": crack_type.value,
            "noise_class": noise_class.value,
            "is_crack": 1 if source_class in [SourceClass.CRACK_INITIATION, SourceClass.CRACK_GROWTH, SourceClass.CRACK_FRICTION] else 0,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "cell_id": cell_id,
            "temperature_c": temperature,
            "energy": energy,
            "triggered": 1 if daq_out["triggered"] else 0,
            "peak_amp_mv": feats.peak_amp_mv,
            "rise_time_us": feats.rise_time_us,
            "duration_us": feats.duration_us,
            "counts": feats.counts,
            "energy_au": feats.energy_au,
            "centroid_khz": feats.centroid_khz,
            "peak_freq_khz": feats.peak_freq_khz,
            "config_hash": self.config_hash,
            "feature_vector": feat_vector,
        }

        return calibrated_volts, meta

    def generate_batch(
        self,
        n_events: int = 100,
        crack_ratio: float = 0.6,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate a batch of balanced synthetic events.
        Returns:
            (waveforms: np.ndarray shape (n_events, 4, n_samples), dataframe_index)
        """
        rng = np.random.default_rng(self.seed)
        all_waves = []
        all_meta = []
        
        crack_types = [CrackType.DIAGONAL, CrackType.DENDRITIC, CrackType.PARALLEL_TO_BUSBAR, CrackType.EDGE_CHIPPING]
        noise_classes = [NoiseClass.RAIN_HAIL, NoiseClass.WIND_RATTLE, NoiseClass.THERMAL_TICK, NoiseClass.EMI_BURST, NoiseClass.HANDLING_TAP]

        for i in range(n_events):
            is_crack = bool(rng.random() < crack_ratio)
            if is_crack:
                src_idx = int(rng.integers(0, 3))
                src_cls = [SourceClass.CRACK_INITIATION, SourceClass.CRACK_GROWTH, SourceClass.CRACK_FRICTION][src_idx]
                c_idx = int(rng.integers(0, len(crack_types)))
                c_type = crack_types[c_idx]
                n_type = NoiseClass.NONE
            else:
                src_cls = SourceClass.NOISE
                c_type = CrackType.NONE
                n_idx = int(rng.integers(0, len(noise_classes)))
                n_type = noise_classes[n_idx]
                
            waves, meta = self.generate_event(
                event_id=i,
                source_class=src_cls,
                crack_type=c_type,
                noise_class=n_type,
                rng=rng,
            )
            all_waves.append(waves)
            all_meta.append(meta)

        waveforms_array = np.stack(all_waves, axis=0) # shape (N, 4, n_samples)
        df_index = pd.DataFrame(all_meta)
        return waveforms_array, df_index
