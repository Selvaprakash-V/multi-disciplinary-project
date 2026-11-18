"""
AcoustiSolar live demo server (FastAPI).

Simulates an event with the Tier-0 twin and DAQ twin, localises it with the gated TDOA
pipeline and classifies it with the trained feature model.

Integrity rules (docs/research/CODEBASE_EVALUATION.md §2.4):
* the model is loaded only if its SHA-256 matches the sidecar file written at training;
* if no verified model is available the API says so — it never echoes the ground truth as
  a "prediction";
* ``/api/benchmark_summary`` serves the computed benchmark / verification JSON files, not
  hard-coded numbers.
"""

import base64
import json
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

root = Path(__file__).resolve().parent.parent
for pkg in ("schemas", "physics", "daqtwin", "dsp", "ml", "sdk"):
    sys.path.insert(0, str(root / "packages" / pkg / "src"))
sys.path.insert(0, str(root))

from demo.image_analyzer import analyze_solar_image, generate_preset_cell

from acoustisolar_daqtwin.engine import DAQDigitalTwin  # noqa: E402
from acoustisolar_dsp.features import AEFeatureExtractor  # noqa: E402
from acoustisolar_dsp.pickers import OnsetPicker  # noqa: E402
from acoustisolar_dsp.tdoa import TDOALocalizer  # noqa: E402
from acoustisolar_ml.baseline_model import AEFeatureClassifier  # noqa: E402
from acoustisolar_physics.tier0 import Tier0Simulator  # noqa: E402
from acoustisolar_schemas.daq import DAQConfig  # noqa: E402
from acoustisolar_schemas.event import NEW_FRACTURE_CLASSES  # noqa: E402
from acoustisolar_schemas.module import ModuleTwinConfig  # noqa: E402
from acoustisolar_schemas.sensor import SensorArrayConfig  # noqa: E402
from acoustisolar_schemas.source import CrackType, NoiseClass, SourceClass, SourceEventConfig  # noqa: E402

app = FastAPI(title="AcoustiSolar Live Simulation Engine", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

module_cfg = ModuleTwinConfig(width_mm=1040.0, length_mm=1760.0)
sensor_cfg = SensorArrayConfig.for_module(module_cfg.width_mm, module_cfg.length_mm)
daq_cfg = DAQConfig()
sim = Tier0Simulator(module_config=module_cfg, sensor_config=sensor_cfg)
daq_twin = DAQDigitalTwin(daq_config=daq_cfg, sensor_config=sensor_cfg, seed=42)
feat_extractor = AEFeatureExtractor(sample_rate_hz=daq_cfg.adc.rate_hz)
localizer = TDOALocalizer(sensor_cfg.positions_mm)
_sim_lock = threading.Lock()

MODEL_PATH = root / "models" / "ae_feature_classifier.joblib"
clf = None
model_status = "missing"
if MODEL_PATH.exists():
    try:
        clf = AEFeatureClassifier.load(MODEL_PATH)  # verifies SHA-256 sidecar before loading
        model_status = "verified"
    except Exception as exc:  # refuse unverified / unloadable models
        model_status = f"refused: {exc}"

SOURCES = {
    "crack_initiation": (SourceClass.CRACK_INITIATION, CrackType.DIAGONAL, NoiseClass.NONE, 1),
    "crack_growth": (SourceClass.CRACK_GROWTH, CrackType.PARALLEL_TO_BUSBAR, NoiseClass.NONE, 1),
    "crack_friction": (SourceClass.CRACK_FRICTION, CrackType.DIAGONAL, NoiseClass.NONE, 0),
    "noise_rain": (SourceClass.NOISE, CrackType.NONE, NoiseClass.RAIN, 0),
    "noise_hail": (SourceClass.NOISE, CrackType.NONE, NoiseClass.HAIL, 0),
    "noise_inverter": (SourceClass.NOISE, CrackType.NONE, NoiseClass.EMI_BURST, 0),
    "noise_wind": (SourceClass.NOISE, CrackType.NONE, NoiseClass.WIND_RATTLE, 0),
    "noise_thermal": (SourceClass.NOISE, CrackType.NONE, NoiseClass.THERMAL_TICK, 0),
}


# Label the classifier is expected to return for each demo source type.
SOURCE_TO_EVENT_CLASS = {
    "crack_initiation": "crack_initiation", "crack_growth": "crack_growth", "crack_friction": "crack_friction",
    "noise_rain": "rain", "noise_hail": "hail", "noise_inverter": "emi_burst",
    "noise_wind": "wind_rattle", "noise_thermal": "thermal_tick",
}
SOURCE_ONSET_US = 400.0  # emission time of the simulated source inside the 2.56 ms record


class SimulationRequest(BaseModel):
    x_mm: float
    y_mm: float
    source_type: str = "crack_initiation"
    temperature_c: float = 25.0
    energy: float = 1.5
    seed: Optional[int] = None  # same seed + same inputs -> identical event


def get_cell_coordinates(x_mm, y_mm, width_mm, length_mm, rows=20, cols=6):
    col = int(min(max(0, (x_mm / width_mm) * cols), cols - 1))
    row = int(min(max(0, (y_mm / length_mm) * rows), rows - 1))
    return row, col, f"Cell (R{row + 1}, C{col + 1})"


@app.get("/")
def get_index():
    index_file = Path(__file__).parent / "static" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>AcoustiSolar Demo Engine Online</h1>")


@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    seed = int(req.seed) if req.seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
    with _sim_lock:  # the DAQ twin is shared; keep its noise streams per request
        return _simulate(req, seed)


def _simulate(req: SimulationRequest, seed: int):
    rng = np.random.default_rng(seed)
    # Electronic noise (preamp, ADC jitter) is drawn from generators inside the DAQ twin.
    # Re-seed them from the request seed so the same seed reproduces the same record;
    # the sensor gain spread is fixed at start-up and represents the installed hardware.
    daq_twin.analog_fe.rng = np.random.default_rng([seed, 1])
    daq_twin.adc.rng = np.random.default_rng([seed, 2])
    st = req.source_type.lower()
    sc, ct, nc, is_crack_gt = SOURCES.get(st, SOURCES["crack_initiation"])
    width, length = module_cfg.width_mm, module_cfg.length_mm
    x = float(np.clip(req.x_mm, 0.0, width))
    y = float(np.clip(req.y_mm, 0.0, length))
    row_gt, col_gt, cell_label_gt = get_cell_coordinates(x, y, width, length)

    src_cfg = SourceEventConfig(source_class=sc, crack_type=ct, noise_class=nc, x_mm=x, y_mm=y,
                                energy_arbitrary=float(np.clip(req.energy, 0.05, 50.0)),
                                rise_time_us=float(rng.uniform(1.0, 3.0)))
    temp = float(np.clip(req.temperature_c, -30.0, 90.0))
    mech = sim.simulate(src_cfg, temperature_c=temp, rng=rng, onset_s=SOURCE_ONSET_US * 1e-6)
    mech = mech + sim.source_lib.ambient_background(mech.shape[0], rng, level=2e-7)
    daq_out = daq_twin.process_mechanical_waves(mech, channel_skew_ns=rng.uniform(-5.0, 5.0, 4))
    volts = daq_out["calibrated_volts"]

    onset_samples = [int(OnsetPicker.threshold_onset(volts[ch])) for ch in range(4)]
    v_s0 = float(sim.s0_velocity(temp))
    v_a0 = float(sim.a0_front_velocity(temp))
    loc = localizer.locate(volts, v_eff_m_s=v_s0, v_a0_m_s=v_a0, min_snr_db=10.0, bounds_mm=(0.0, width, 0.0, length))
    x_est = float(np.clip(loc["x_mm"], 0.0, width)) if np.isfinite(loc["x_mm"]) else float("nan")
    y_est = float(np.clip(loc["y_mm"], 0.0, length)) if np.isfinite(loc["y_mm"]) else float("nan")
    located = np.isfinite(x_est)
    loc_err = float(np.hypot(x_est - x, y_est - y)) if located else float("nan")
    row_est, col_est, cell_label_est = get_cell_coordinates(x_est, y_est, width, length) if located else (-1, -1, "n/a")
    arr = np.array([a if a is not None else np.nan for a in loc["arrival_samples"]], dtype=float)
    dt_us = [
        round(float(a - arr[0]), 2) if (np.isfinite(a) and np.isfinite(arr[0])) else None
        for a in arr
    ]

    features = feat_extractor.extract_channel_features(volts, (x_est, y_est) if located else None, (width, length), temp)
    ch1 = feat_extractor.extract_features(volts[0])
    if clf is not None:
        proba = clf.predict_proba(features[None, :])[0]
        p_crack = float(sum(proba[i] for i, c in enumerate(clf.classes) if c in NEW_FRACTURE_CLASSES))
        pred_class = clf.classes[int(np.argmax(proba))]
        pred = int(p_crack >= 0.5)
        ml = {
            "is_crack_pred": pred,
            "verdict": "CRACK DETECTED" if pred else "NO NEW FRACTURE",
            "predicted_source": pred_class,
            "crack_probability": round(p_crack * 100.0, 1),
            "confidence": round(float(np.max(proba)) * 100.0, 1),
            "model_name": "AEFeatureClassifier (gradient-boosted trees, SHA-256 verified)",
            "class_probabilities": {c: round(float(p), 4) for c, p in zip(clf.classes, proba)},
            "new_fracture_classes": sorted(NEW_FRACTURE_CLASSES),
            "decision_threshold": 0.5,
        }
    else:
        ml = {"is_crack_pred": -1, "verdict": "MODEL UNAVAILABLE", "predicted_source": None,
              "crack_probability": 0.0, "confidence": 0.0, "model_name": f"none ({model_status})",
              "class_probabilities": {}, "new_fracture_classes": sorted(NEW_FRACTURE_CLASSES),
              "decision_threshold": 0.5}

    fs = float(daq_cfg.adc.rate_hz)
    arrival_us = [None if not np.isfinite(a) else round(float(a) / fs * 1e6, 2) for a in arr]

    def fin(v, nd=1):
        return round(float(v), nd) if v is not None and np.isfinite(v) else None

    # Oscilloscope "peak detect": keep the largest-magnitude sample of each block of `step`
    # samples (sign preserved). Plain decimation (every 10th sample = 100 kS/s) would alias
    # the 100-300 kHz AE burst and hide its envelope.
    step = 10
    n_blk = volts.shape[1] // step
    blocks = volts[:, : n_blk * step].reshape(volts.shape[0], n_blk, step)
    pick = np.abs(blocks).argmax(axis=2)
    peak_detect = np.take_along_axis(blocks, pick[..., None], axis=2)[..., 0] * 1000.0  # mV
    return {
        "run": {"seed": seed, "temperature_c": temp, "energy": float(src_cfg.energy_arbitrary),
                "source_type": st, "s0_velocity_m_s": round(v_s0, 1), "source_onset_us": SOURCE_ONSET_US,
                "sample_rate_hz": fs, "adc_bits": int(daq_cfg.adc.bits), "triggered": bool(daq_out["triggered"]),
                "model_status": model_status},
        "ground_truth": {"x_mm": round(x, 1), "y_mm": round(y, 1), "cell_row": row_gt, "cell_col": col_gt,
                         "cell_id": cell_label_gt, "is_crack": is_crack_gt, "source_type": st,
                         "event_class": SOURCE_TO_EVENT_CLASS.get(st, st)},
        "reconstruction": {
            "x_mm": round(x_est, 1) if located else None, "y_mm": round(y_est, 1) if located else None,
            "cell_row": row_est, "cell_col": col_est, "cell_id": cell_label_est,
            "cell_matched": bool(located and row_gt == row_est and col_gt == col_est),
            "error_mm": round(loc_err, 1) if located else None,
            "time_differences_us": dt_us,
            "uncertainty_semi_major_mm": round(float(loc["uncertainty_semi_major_mm"]), 1) if located else None,
            "uncertainty_semi_minor_mm": fin(loc.get("uncertainty_semi_minor_mm")) if located else None,
            "uncertainty_angle_deg": fin(loc.get("uncertainty_angle_deg")) if located else None,
            "residual_rms_us": fin(loc.get("residual_rms_s", float("nan")) * 1e6, 2) if located else None,
            "sensors_used": int(loc.get("sensors_used", 0)),
            "arrival_us": arrival_us,
        },
        "ml_inference": ml,
        "features": {
            "peak_amp_mv": round(ch1.peak_amp_mv, 2), "rise_time_us": round(ch1.rise_time_us, 1),
            "duration_us": round(ch1.duration_us, 1), "counts": ch1.counts, "energy_au": round(ch1.energy_au, 2),
            "centroid_khz": round(ch1.centroid_khz, 1), "peak_freq_khz": round(ch1.peak_freq_khz, 1),
        },
        "oscilloscope": {
            "time_us": (np.arange(n_blk) * step * 1.0).tolist(),
            "waveforms": {f"ch{ch + 1}": peak_detect[ch].round(3).tolist() for ch in range(4)},
            "display_mode": f"peak detect, {step} samples per point",
            "onset_samples": onset_samples,
            "pre_trigger_samples": int(SOURCE_ONSET_US * fs / 1e6),
            # full-resolution record (1 sample = 1 us at 1 MS/s) for plotting, spectra and export
            "samples_mv": [(volts[ch] * 1000.0).round(3).tolist() for ch in range(volts.shape[0])],
        },
        "module": {
            "width_mm": width, "length_mm": length, "rows": 20, "cols": 6,
            "sensors": [{"name": f"S{i + 1}", "x_mm": p[0], "y_mm": p[1]} for i, p in enumerate(sensor_cfg.positions_mm)],
        },
    }


class ImageAnalysisRequest(BaseModel):
    image_base64: str


@app.post("/api/analyze-image")
def analyze_image_endpoint(req: ImageAnalysisRequest):
    data = req.image_base64
    if "," in data:
        data = data.split(",", 1)[1]
    raw_bytes = base64.b64decode(data)
    return analyze_solar_image(raw_bytes)


@app.post("/api/upload-image")
async def upload_image_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    return analyze_solar_image(contents)


@app.get("/api/preset-image/{preset_name}")
def get_preset_image_endpoint(preset_name: str):
    img = generate_preset_cell(preset_name)
    _, buf = cv2.imencode(".png", img)
    return analyze_solar_image(buf.tobytes())


@app.get("/api/benchmark_summary")
def get_benchmark_summary():
    """Computed results only; fields are null when the producing script has not been run."""
    def load(rel):
        p = root / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    return {
        "kpi_benchmark": load("validation/kpi_benchmark_summary.json"),
        "physics_verification": load("validation/physics_verification_results.json"),
        "vv_suite": load("validation/out/vv_suite_results.json"),
        "reproducibility": load("data/reproducibility_proof.json"),
        "model_status": model_status,
        "model_sha256": MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".sha256").read_text().strip().split()[0]
        if MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".sha256").exists() else None,
        "caveat": "Simulation-only results; no real PV acoustic-emission data have been used yet.",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
