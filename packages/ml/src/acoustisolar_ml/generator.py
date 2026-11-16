"""
Synthetic dataset generation (Tier-0 physics -> DAQ twin -> DSP features/localisation).

Labels
------
``event_class`` is one of ``acoustisolar_schemas.event.EVENT_CLASSES``.  ``is_crack`` now
means **new fracture** (crack initiation, crack growth, hail-induced crack); crack friction
(rubbing of an existing crack) is reported separately as ``is_crack_activity``.

Realism added (docs/research/CODEBASE_EVALUATION.md §3, §5.7)
-------------------------------------------------------------
* events start after a random onset so records contain a noise-only pre-trigger segment
* ambient background (broadband + inverter harmonics) at a random level
* physically motivated locations (frame stick-slip and wind rattle at edges / clamps)
* compound "hail_crack" events: a hail impact followed by a nearby crack burst
* domain randomisation: temperature, edge reflection coefficient, sensor installation
  jitter (localiser keeps nominal positions), measured-temperature error, and optional
  material variants (EVA modulus, glass thickness) that each get their own laminate model
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from acoustisolar_daqtwin.engine import DAQDigitalTwin
from acoustisolar_dsp.features import AEFeatureExtractor
from acoustisolar_dsp.tdoa import TDOALocalizer
from acoustisolar_physics.tier0 import Tier0Simulator
from acoustisolar_schemas.daq import DAQConfig
from acoustisolar_schemas.event import CRACK_ACTIVITY_CLASSES, EVENT_CLASSES, NEW_FRACTURE_CLASSES
from acoustisolar_schemas.hashing import compute_config_hash
from acoustisolar_schemas.module import ModuleTwinConfig
from acoustisolar_schemas.sensor import SensorArrayConfig
from acoustisolar_schemas.source import CrackType, NoiseClass, SourceClass, SourceEventConfig

_CLASS_TO_SOURCE = {
    "crack_initiation": (SourceClass.CRACK_INITIATION, NoiseClass.NONE),
    "crack_growth": (SourceClass.CRACK_GROWTH, NoiseClass.NONE),
    "crack_friction": (SourceClass.CRACK_FRICTION, NoiseClass.NONE),
    "hail": (SourceClass.NOISE, NoiseClass.HAIL),
    "rain": (SourceClass.NOISE, NoiseClass.RAIN),
    "wind_rattle": (SourceClass.NOISE, NoiseClass.WIND_RATTLE),
    "thermal_tick": (SourceClass.NOISE, NoiseClass.THERMAL_TICK),
    "emi_burst": (SourceClass.NOISE, NoiseClass.EMI_BURST),
    "handling_tap": (SourceClass.NOISE, NoiseClass.HANDLING_TAP),
}
_CRACK_TYPES = [CrackType.DIAGONAL, CrackType.DENDRITIC, CrackType.PARALLEL_TO_BUSBAR, CrackType.EDGE_CHIPPING]


def class_probabilities(crack_ratio: float = 0.6) -> Dict[str, float]:
    """Default class mix: ``crack_ratio`` of events are crack-family."""
    crack = {"crack_initiation": 0.35, "crack_growth": 0.35, "hail_crack": 0.15, "crack_friction": 0.15}
    noise = ["hail", "rain", "wind_rattle", "thermal_tick", "emi_burst", "handling_tap"]
    p = {k: v * crack_ratio for k, v in crack.items()}
    p.update({k: (1.0 - crack_ratio) / len(noise) for k in noise})
    return p


def perturbed_module(module: ModuleTwinConfig, rng: np.random.Generator) -> ModuleTwinConfig:
    """Material variant: EVA modulus x U(0.6, 1.6), glass thickness +/- 0.15 mm."""
    m = module.model_copy(deep=True)
    eva_scale = float(rng.uniform(0.6, 1.6))
    for layer in m.stack:
        if layer.material_model == "viscoelastic_prony":
            layer.youngs_modulus_gpa *= eva_scale
    m.stack[0].thickness_mm = float(np.clip(m.stack[0].thickness_mm + rng.uniform(-0.15, 0.15), 1.6, 4.5))
    return m


class DatasetGenerator:
    def __init__(
        self,
        module_config: Optional[ModuleTwinConfig] = None,
        sensor_config: Optional[SensorArrayConfig] = None,
        daq_config: Optional[DAQConfig] = None,
        seed: int = 42,
        temperature_range_c: Tuple[float, float] = (-10.0, 65.0),
        onset_range_us: Tuple[float, float] = (300.0, 450.0),
        ambient_noise: bool = True,
        n_material_variants: int = 1,
        sensor_jitter_mm: float = 3.0,
        localize: bool = True,
    ):
        self.module_config = module_config or ModuleTwinConfig()
        self.sensor_config = sensor_config or SensorArrayConfig.for_module(
            self.module_config.width_mm, self.module_config.length_mm
        )
        self.daq_config = daq_config or DAQConfig()
        self.seed = seed
        self.temperature_range_c = temperature_range_c
        self.onset_range_us = onset_range_us
        self.ambient_noise = ambient_noise
        self.sensor_jitter_mm = sensor_jitter_mm
        self.localize = localize
        self.config_hash = compute_config_hash(self.module_config)
        self.feature_extractor = AEFeatureExtractor(sample_rate_hz=self.daq_config.adc.rate_hz)

        fs = self.daq_config.adc.rate_hz
        n = self.daq_config.trigger.total_samples
        variant_rng = np.random.default_rng(seed + 7777)
        modules = [self.module_config] + [
            perturbed_module(self.module_config, variant_rng) for _ in range(max(0, n_material_variants - 1))
        ]
        self.sims = [Tier0Simulator(m, self.sensor_config, sample_rate_hz=fs, n_samples=n) for m in modules]
        self.sim = self.sims[0]
        self.daq_twin = DAQDigitalTwin(daq_config=self.daq_config, sensor_config=self.sensor_config, seed=seed)
        self.localizer = TDOALocalizer(self.sensor_config.positions_mm, sample_rate_hz=fs)

    # ------------------------------------------------------------------ geometry
    def _random_location(self, event_class: str, rng: np.random.Generator) -> Tuple[float, float]:
        W, L = self.module_config.width_mm, self.module_config.length_mm
        if event_class == "thermal_tick":
            d = rng.uniform(5.0, 25.0)
            edge = int(rng.integers(0, 4))
            along = rng.uniform(0.05, 0.95)
            return [(d, along * L), (W - d, along * L), (along * W, d), (along * W, L - d)][edge]
        if event_class == "wind_rattle":
            # mounting clamps at the quarter points of the long edges
            y = L * float(rng.choice([0.25, 0.75])) + rng.uniform(-50.0, 50.0)
            x = rng.uniform(5.0, 25.0) if rng.random() < 0.5 else W - rng.uniform(5.0, 25.0)
            return x, y
        margin = 30.0
        return float(rng.uniform(margin, W - margin)), float(rng.uniform(margin, L - margin))

    def _cell_id(self, x_mm: float, y_mm: float) -> str:
        lay = self.module_config.layout
        return TDOALocalizer.as_cell(x_mm, y_mm, self.module_config.width_mm, self.module_config.length_mm,
                                     lay.rows, lay.cols)

    # ------------------------------------------------------------------ events
    def generate_event(
        self,
        event_id: int,
        source_class: Optional[SourceClass] = None,
        crack_type: CrackType = CrackType.NONE,
        noise_class: NoiseClass = NoiseClass.NONE,
        temp_c: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        event_class: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """One simulated, digitised, localised and featurised event.

        Either pass ``event_class`` (preferred) or the legacy (source_class, noise_class)."""
        if rng is None:
            rng = np.random.default_rng(self.seed + event_id)
        if event_class is None:
            event_class = self._legacy_class(source_class, noise_class)

        temperature = float(temp_c) if temp_c is not None else float(rng.uniform(*self.temperature_range_c))
        sim = self.sims[int(rng.integers(0, len(self.sims)))]
        sim.edge_reflection_coeff = float(rng.uniform(0.3, 0.75))
        onset = float(rng.uniform(*self.onset_range_us)) * 1.0e-6
        jitter = rng.normal(0.0, self.sensor_jitter_mm, size=(self.sensor_config.count, 2))
        true_positions = [tuple(np.add(p, j)) for p, j in zip(self.sensor_config.positions_mm, jitter)]

        x_mm, y_mm = self._random_location("hail" if event_class == "hail_crack" else event_class, rng)
        energy = float(10 ** rng.uniform(-0.75, 0.75))  # ~30 dB spread of source levels
        c_type = _CRACK_TYPES[int(rng.integers(0, len(_CRACK_TYPES)))] if event_class in CRACK_ACTIVITY_CLASSES else CrackType.NONE

        if event_class == "hail_crack":
            hail = SourceEventConfig(source_class=SourceClass.NOISE, noise_class=NoiseClass.HAIL,
                                     x_mm=x_mm, y_mm=y_mm, energy_arbitrary=energy)
            mech = sim.simulate(hail, temperature, rng, onset, true_positions)
            cx = float(np.clip(x_mm + rng.uniform(-60, 60), 20, self.module_config.width_mm - 20))
            cy = float(np.clip(y_mm + rng.uniform(-60, 60), 20, self.module_config.length_mm - 20))
            crack = SourceEventConfig(source_class=SourceClass.CRACK_INITIATION, crack_type=c_type,
                                      x_mm=cx, y_mm=cy, energy_arbitrary=float(10 ** rng.uniform(-0.3, 0.9)),
                                      rise_time_us=float(rng.uniform(0.8, 3.5)))
            mech = mech + sim.simulate(crack, temperature, rng, onset + rng.uniform(20e-6, 400e-6), true_positions)
            src_cls, n_cls, label_xy = SourceClass.CRACK_INITIATION, NoiseClass.HAIL, (cx, cy)
        else:
            src_cls, n_cls = _CLASS_TO_SOURCE[event_class]
            cfg = SourceEventConfig(source_class=src_cls, crack_type=c_type, noise_class=n_cls,
                                    x_mm=x_mm, y_mm=y_mm, energy_arbitrary=energy,
                                    rise_time_us=float(rng.uniform(0.8, 3.5)))
            mech = sim.simulate(cfg, temperature, rng, onset, true_positions)
            label_xy = (x_mm, y_mm)

        if self.ambient_noise:
            level = float(10 ** rng.uniform(np.log10(3e-8), np.log10(1e-6)))
            mech = mech + sim.source_lib.ambient_background(mech.shape[0], rng, level=level)

        skew = rng.uniform(-10.0, 10.0, size=self.sensor_config.count)
        daq_out = self.daq_twin.process_mechanical_waves(mech, channel_skew_ns=skew)
        volts = daq_out["calibrated_volts"]

        measured_temp = temperature + float(rng.normal(0.0, 2.0))
        loc = None
        if self.localize:
            loc = self.localizer.locate(
                volts, v_eff_m_s=sim.s0_velocity(measured_temp),
                bounds_mm=(0.0, self.module_config.width_mm, 0.0, self.module_config.length_mm),
            )
        est_xy = (loc["x_mm"], loc["y_mm"]) if loc and np.isfinite(loc["x_mm"]) else None
        module_size = (self.module_config.width_mm, self.module_config.length_mm)
        feat_vector = self.feature_extractor.extract_channel_features(volts, est_xy, module_size, measured_temp)
        ch0 = self.feature_extractor.extract_features(volts[0])

        meta = {
            "event_id": f"EVT-{self.seed}-{event_id:06d}",
            "module_id": self.module_config.module_id,
            "event_class": event_class,
            "source_class": src_cls.value,
            "crack_type": c_type.value,
            "noise_class": n_cls.value,
            "is_crack": int(event_class in NEW_FRACTURE_CLASSES),
            "is_crack_activity": int(event_class in CRACK_ACTIVITY_CLASSES),
            "x_mm": label_xy[0],
            "y_mm": label_xy[1],
            "cell_id": self._cell_id(*label_xy),
            "temperature_c": temperature,
            "energy": energy,
            "onset_us": onset * 1e6,
            "triggered": int(daq_out["triggered"]),
            "clipped": int(np.max(np.abs(volts)) >= 0.98 * self.daq_config.adc.full_scale_volts / 2.0),
            "peak_amp_mv": ch0.peak_amp_mv,
            "rise_time_us": ch0.rise_time_us,
            "duration_us": ch0.duration_us,
            "counts": ch0.counts,
            "energy_au": ch0.energy_au,
            "centroid_khz": ch0.centroid_khz,
            "peak_freq_khz": ch0.peak_freq_khz,
            "loc_x_mm": est_xy[0] if est_xy else np.nan,
            "loc_y_mm": est_xy[1] if est_xy else np.nan,
            "loc_error_mm": float(np.hypot(est_xy[0] - label_xy[0], est_xy[1] - label_xy[1])) if est_xy else np.nan,
            "loc_cell_id": self._cell_id(*est_xy) if est_xy else "",
            "loc_uncertainty_mm": loc["uncertainty_semi_major_mm"] if loc else np.nan,
            "config_hash": self.config_hash,
            "feature_vector": feat_vector,
        }
        return volts, meta

    @staticmethod
    def _legacy_class(source_class: Optional[SourceClass], noise_class: NoiseClass) -> str:
        if source_class is None or source_class == SourceClass.CRACK_INITIATION:
            return "crack_initiation"
        if source_class == SourceClass.CRACK_GROWTH:
            return "crack_growth"
        if source_class == SourceClass.CRACK_FRICTION:
            return "crack_friction"
        mapping = {
            NoiseClass.HAIL: "hail", NoiseClass.RAIN_HAIL: "hail", NoiseClass.RAIN: "rain",
            NoiseClass.WIND_RATTLE: "wind_rattle", NoiseClass.THERMAL_TICK: "thermal_tick",
            NoiseClass.EMI_BURST: "emi_burst", NoiseClass.HANDLING_TAP: "handling_tap",
        }
        return mapping.get(noise_class, "handling_tap")

    def generate_batch(
        self,
        n_events: int = 100,
        crack_ratio: float = 0.6,
        class_probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Batch of events: (waveforms (N, n_channels, n_samples), metadata DataFrame)."""
        rng = np.random.default_rng(self.seed)
        probs = class_probs or class_probabilities(crack_ratio)
        names = [c for c in EVENT_CLASSES if c in probs]
        p = np.array([probs[c] for c in names], dtype=float)
        p /= p.sum()
        waves: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        for i in range(n_events):
            cls = names[int(rng.choice(len(names), p=p))]
            w, m = self.generate_event(i, rng=rng, event_class=cls)
            waves.append(w)
            metas.append(m)
        return np.stack(waves, axis=0), pd.DataFrame(metas)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["feature_vector"].values).astype(np.float64)


def label_vector(df: pd.DataFrame, classes: Sequence[str] = EVENT_CLASSES) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return df["event_class"].map(index).values.astype(int)
