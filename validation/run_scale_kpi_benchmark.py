"""
Scale KPI benchmark with strict module-level splits (PRD §9.5, §14.1).

* 5 physically distinct module designs; train on MOD-1..3, test on unseen MOD-4/5.
  Tier-0 consumes each module's full stack (glass, EVA/POE viscoelasticity, size) and
  sensors sit at each module's corners.  MOD-5 runs hot (55-75 C).
* Models: gradient-boosted trees on engineered features (+ temperature scaling, split
  conformal sets) and a trainable 1D residual CNN on waveforms (if PyTorch is available).
* Open-set test: the classifier and the Mahalanobis "unknown source" detector are trained
  without one source type, which then appears only at test time.
* Detection: untriggered crack events count as missed detections.  Recall at <= 1 false
  alarm per module-day uses a Generalised-Pareto tail fit on noise-event scores; the
  trigger-rate assumption (noise triggers per module-day) is [UNSOURCED].
* Localisation on held-out crack events; Tier-1 localisation with a correct setup
  (reflecting frame edges, sensors outside any absorber, >= 10 points per wavelength).

Outputs: validation/kpi_benchmark_summary.json (tracked), models/ae_feature_classifier.joblib
(+ .sha256), models/waveform_resnet.pt, data/benchmark_20k/* (git-ignored raw data).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
for pkg in ("schemas", "physics", "daqtwin", "dsp", "ml", "sdk"):
    sys.path.insert(0, str(root / "packages" / pkg / "src"))

from acoustisolar_dsp.tdoa import TDOALocalizer  # noqa: E402
from acoustisolar_ml.baseline_model import AEFeatureClassifier  # noqa: E402
from acoustisolar_ml.calibration import ConformalClassifier, TemperatureScaler  # noqa: E402
from acoustisolar_ml.generator import DatasetGenerator, feature_matrix, label_vector  # noqa: E402
from acoustisolar_ml.metrics import (  # noqa: E402
    bootstrap_ci,
    compute_classification_metrics,
    compute_localization_metrics,
    evt_threshold,
    multiclass_ece,
    multiclass_metrics,
)
from acoustisolar_ml.ood import MahalanobisOOD  # noqa: E402
from acoustisolar_physics.tier1 import Tier1WaveSolver  # noqa: E402
from acoustisolar_schemas.event import EVENT_CLASSES  # noqa: E402
from acoustisolar_schemas.module import LayerConfig, ModuleLayoutConfig, ModuleTwinConfig  # noqa: E402

NOISE_TRIGGERS_PER_MODULE_DAY = 10_000  # [UNSOURCED] — replace with field logging (ASSUMPTIONS §4A-3)
TARGETS = {"recall_at_1_far": 0.98, "macro_f1": 0.95, "loc_median_mm": 15.0, "loc_p90_mm": 40.0,
           "cell_id_accuracy": 0.95, "ece": 0.05}


def _visco(name, t, rho, e_gpa, nu, temp=25.0):
    return LayerConfig(name=name, thickness_mm=t, density_kg_m3=rho, youngs_modulus_gpa=e_gpa, poissons_ratio=nu,
                       material_model="viscoelastic_prony", damping_ratio=0.15, temperature_c=temp)


def module_configs():
    mod1 = ModuleTwinConfig(module_id="MOD-1-STD120", width_mm=1040.0, length_mm=1760.0)
    mod2 = ModuleTwinConfig(
        module_id="MOD-2-LARGE144", width_mm=1134.0, length_mm=2278.0,
        layout=ModuleLayoutConfig(cells=144, rows=24, cols=6, cell_size="G12"),
        stack=[LayerConfig(name="glass", thickness_mm=2.8, density_kg_m3=2500.0, youngs_modulus_gpa=74.0, poissons_ratio=0.21),
               _visco("eva_front", 0.50, 960.0, 0.015, 0.45),
               LayerConfig(name="si_cell", thickness_mm=0.150, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
               _visco("eva_back", 0.50, 960.0, 0.015, 0.45),
               LayerConfig(name="backsheet", thickness_mm=0.35, density_kg_m3=1200.0, youngs_modulus_gpa=3.8, poissons_ratio=0.35)])
    mod3 = ModuleTwinConfig(
        module_id="MOD-3-UTIL60", width_mm=992.0, length_mm=1650.0,
        layout=ModuleLayoutConfig(cells=60, half_cut=False, rows=10, cols=6, cell_size="M6"),
        stack=[LayerConfig(name="glass", thickness_mm=4.0, density_kg_m3=2500.0, youngs_modulus_gpa=72.0, poissons_ratio=0.22),
               _visco("eva_front", 0.45, 960.0, 0.0125, 0.45),
               LayerConfig(name="si_cell", thickness_mm=0.200, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
               _visco("eva_back", 0.45, 960.0, 0.0125, 0.45),
               LayerConfig(name="backsheet", thickness_mm=0.30, density_kg_m3=1200.0, youngs_modulus_gpa=3.5, poissons_ratio=0.35)])
    mod4 = ModuleTwinConfig(
        module_id="MOD-4-BIFACIAL72", width_mm=1000.0, length_mm=2000.0,
        layout=ModuleLayoutConfig(cells=72, half_cut=False, rows=12, cols=6, cell_size="M10"),
        stack=[LayerConfig(name="glass_front", thickness_mm=2.0, density_kg_m3=2500.0, youngs_modulus_gpa=72.0, poissons_ratio=0.22),
               _visco("poe_front", 0.40, 920.0, 0.020, 0.43),
               LayerConfig(name="si_cell", thickness_mm=0.160, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
               _visco("poe_back", 0.40, 920.0, 0.020, 0.43),
               LayerConfig(name="glass_back", thickness_mm=2.0, density_kg_m3=2500.0, youngs_modulus_gpa=72.0, poissons_ratio=0.22)])
    mod5 = ModuleTwinConfig(
        module_id="MOD-5-HOT65C", width_mm=1040.0, length_mm=1760.0,
        stack=[LayerConfig(name="glass", thickness_mm=3.2, density_kg_m3=2500.0, youngs_modulus_gpa=70.0, poissons_ratio=0.23),
               _visco("eva_front", 0.45, 960.0, 0.005, 0.48, temp=65.0),
               LayerConfig(name="si_cell", thickness_mm=0.175, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
               _visco("eva_back", 0.45, 960.0, 0.005, 0.48, temp=65.0),
               LayerConfig(name="backsheet", thickness_mm=0.30, density_kg_m3=1200.0, youngs_modulus_gpa=3.2, poissons_ratio=0.36)])
    temps = {"MOD-5-HOT65C": (55.0, 75.0)}
    return [mod1, mod2, mod3], [mod4, mod5], temps


def generate(configs, n_per, temps, seed0):
    waves, dfs = [], []
    for i, cfg in enumerate(configs):
        t = time.time()
        gen = DatasetGenerator(module_config=cfg, seed=seed0 + 100 * i,
                               temperature_range_c=temps.get(cfg.module_id, (-10.0, 65.0)), n_material_variants=2)
        w, df = gen.generate_batch(n_events=n_per)
        print(f"  {cfg.module_id}: {n_per} events in {time.time() - t:.0f} s", flush=True)
        waves.append(w.astype(np.float32))  # halve memory; ADC data are 12-bit anyway
        del gen
        dfs.append(df)
    return np.concatenate(waves), pd.concat(dfs, ignore_index=True)


def tier1_localization():
    """Tier-1 (scalar FD) localisation with a correct setup: reflecting frame edges,
    5 mm grid (>= 10 points per wavelength at 100 kHz), 1 MHz read-out."""
    W, L, dx, dt, c = 1.04, 1.76, 0.005, 0.5e-6, 5490.0
    solver = Tier1WaveSolver(width_m=W, length_m=L, dx_m=dx, dt_s=dt, c_speed_m_s=c, boundary="reflecting")
    sensors_m = [(0.06, 0.06), (0.06, L - 0.06), (W - 0.06, 0.06), (W - 0.06, L - 0.06)]
    loc = TDOALocalizer([(x * 1e3, y * 1e3) for x, y in sensors_m])
    n = 1400
    t = np.arange(n) * dt
    f0 = 100e3
    tau = t - 300e-6
    src = (1 - 2 * (np.pi * f0 * tau) ** 2) * np.exp(-((np.pi * f0 * tau) ** 2))
    rng = np.random.default_rng(0)
    errs = []
    for sx, sy in [(0.30, 0.40), (0.50, 0.80), (0.70, 1.20), (0.25, 1.30), (0.80, 0.50), (0.45, 0.95)]:
        w = solver.run_simulation(sx, sy, src, sensors_m)[0][:, ::2]  # decimate to 1 MHz
        w = w / np.max(np.abs(w)) + rng.normal(0, 1e-3, w.shape)
        r = loc.locate(w, v_eff_m_s=c, bounds_mm=(0, W * 1e3, 0, L * 1e3))
        errs.append(float(np.hypot(r["x_mm"] - sx * 1e3, r["y_mm"] - sy * 1e3)))
    return {"n_events": len(errs), "median_err_mm": float(np.median(errs)), "p90_err_mm": float(np.percentile(errs, 90)),
            "errors_mm": errs, "setup": "reflecting edges, dx=5 mm, f0=100 kHz Ricker, gated TDOA"}


def main(n_per_module: int = 1200, out_dir: str = "data/benchmark_20k", seed: int = 1000):
    t_start = time.time()
    out = root / out_dir
    out.mkdir(parents=True, exist_ok=True)
    train_cfgs, test_cfgs, temps = module_configs()
    print("=== Generating training modules ===")
    w_tr, df_tr = generate(train_cfgs, n_per_module, temps, seed)
    print("=== Generating held-out modules ===")
    w_te, df_te = generate(test_cfgs, n_per_module, temps, seed + 5000)

    pd.concat([df_tr.assign(split="train"), df_te.assign(split="test")]).drop(columns=["feature_vector"]).to_parquet(out / "events.parquet")
    np.savez_compressed(out / "waveforms.npz", train=w_tr.astype(np.float32), test=w_te.astype(np.float32))

    X_tr, y_tr = feature_matrix(df_tr), label_vector(df_tr)
    X_te, y_te = feature_matrix(df_te), label_vector(df_te)
    rng = np.random.default_rng(seed)
    cal_mask = rng.random(len(y_tr)) < 0.15
    fit_mask = ~cal_mask

    # ---------------- feature model
    print("=== Training feature classifier ===")
    clf = AEFeatureClassifier(max_iter=400).fit(X_tr[fit_mask], y_tr[fit_mask])
    ts = TemperatureScaler().fit(clf.predict_proba(X_tr[cal_mask]), y_tr[cal_mask])
    P_te = ts.transform(clf.predict_proba(X_te))
    conformal = ConformalClassifier(alpha=0.1).fit(ts.transform(clf.predict_proba(X_tr[cal_mask])), y_tr[cal_mask])
    sets = conformal.predict_sets(P_te)
    mc = multiclass_metrics(y_te, P_te.argmax(1), EVENT_CLASSES)
    ci = bootstrap_ci(lambda a, b: multiclass_metrics(a, b, EVENT_CLASSES)["macro_f1"], y_te, P_te.argmax(1), n_boot=300)

    # ---------------- binary new-fracture detection (system level)
    p_crack = clf.new_fracture_probability(P_te)
    is_crack = df_te["is_crack"].values
    triggered = df_te["triggered"].values.astype(bool)
    pred = (p_crack >= 0.5) & triggered
    binm = compute_classification_metrics(is_crack, pred.astype(int), p_crack)
    noise_scores = p_crack[(is_crack == 0) & triggered]
    target_fpr = 1.0 / NOISE_TRIGGERS_PER_MODULE_DAY
    evt = evt_threshold(noise_scores, target_fpr)
    thr = evt["threshold"]
    crack_detected = triggered & (p_crack >= thr)
    recall_at_far = float(np.mean(crack_detected[is_crack == 1]))

    # ---------------- open-set test (withhold one source type)
    withheld = "handling_tap"
    k = EVENT_CLASSES.index(withheld)
    keep = y_tr != k
    clf_os = AEFeatureClassifier(max_iter=300).fit(X_tr[keep], y_tr[keep])
    ood = MahalanobisOOD(quantile=0.99).fit(X_tr[keep], y_tr[keep])
    unknown_flag = ood.is_unknown(X_te)
    open_set = {
        "withheld_class": withheld,
        "unknown_detection_rate_on_withheld": float(np.mean(unknown_flag[y_te == k])),
        "false_unknown_rate_on_known": float(np.mean(unknown_flag[y_te != k])),
        "withheld_called_new_fracture_rate": float(np.mean(clf_os.new_fracture_probability(clf_os.predict_proba(X_te[y_te == k])) >= 0.5)),
    }

    # ---------------- deep model
    deep = None
    try:
        from acoustisolar_ml.deep_model import Waveform1DResNet

        print("=== Training 1D residual CNN ===")
        net = Waveform1DResNet(seed=seed).fit(w_tr[fit_mask], y_tr[fit_mask], epochs=25, batch_size=64)
        ts_d = TemperatureScaler().fit(net.predict_proba(w_tr[cal_mask]), y_tr[cal_mask])
        Pd = ts_d.transform(net.predict_proba(w_te))
        md = multiclass_metrics(y_te, Pd.argmax(1), EVENT_CLASSES)
        deep = {"macro_f1": md["macro_f1"], "accuracy": md["accuracy"], "ece_top_label": multiclass_ece(y_te, Pd),
                "per_class_f1": {c: v["f1"] for c, v in md["per_class"].items()}, "device": net.device}
        net.save(root / "models" / "waveform_resnet.pt")
        # simple ensemble
        Pe = 0.5 * (P_te + Pd)
        deep["ensemble_macro_f1"] = multiclass_metrics(y_te, Pe.argmax(1), EVENT_CLASSES)["macro_f1"]
    except ImportError:
        deep = {"skipped": "PyTorch not installed"}

    # ---------------- localisation
    loc_rows = df_te[(df_te["is_crack"] == 1) & (df_te["triggered"] == 1) & np.isfinite(df_te["loc_x_mm"])]
    loc = compute_localization_metrics(loc_rows[["x_mm", "y_mm"]].values, loc_rows[["loc_x_mm", "loc_y_mm"]].values,
                                       list(loc_rows["cell_id"]), list(loc_rows["loc_cell_id"]))
    loc_by_module = {}
    for mid, g in loc_rows.groupby("module_id"):
        loc_by_module[mid] = compute_localization_metrics(g[["x_mm", "y_mm"]].values, g[["loc_x_mm", "loc_y_mm"]].values,
                                                          list(g["cell_id"]), list(g["loc_cell_id"]))
    print("=== Tier-1 localisation ===")
    t1 = tier1_localization()

    clf_path = root / "models" / "ae_feature_classifier.joblib"
    digest = clf.save(clf_path)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t_start, 1),
        "dataset": {
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "n_per_module": n_per_module,
            "train_modules": [c.module_id for c in train_cfgs], "test_modules": [c.module_id for c in test_cfgs],
            "class_counts_test": {c: int(np.sum(y_te == i)) for i, c in enumerate(EVENT_CLASSES)},
            "trigger_rate_by_class_test": df_te.groupby("event_class")["triggered"].mean().round(3).to_dict(),
            "clipped_fraction_test": float(df_te["clipped"].mean()),
        },
        "feature_model": {
            "model": "AEFeatureClassifier (HistGradientBoosting) + temperature scaling",
            "temperature": ts.temperature,
            "macro_f1": mc["macro_f1"], "macro_f1_95ci": list(ci), "accuracy": mc["accuracy"],
            "ece_top_label": multiclass_ece(y_te, P_te),
            "per_class_f1": {c: v["f1"] for c, v in mc["per_class"].items()},
            "confusion_matrix": mc["confusion_matrix"], "classes": EVENT_CLASSES,
            "conformal_alpha": 0.1, "conformal_coverage": ConformalClassifier.coverage(sets, y_te),
            "conformal_mean_set_size": float(np.mean([len(s) for s in sets])),
            "model_file": str(clf_path.relative_to(root)), "model_sha256": digest,
        },
        "new_fracture_detection": {
            "f1_at_0.5": binm["f1_score"], "precision_at_0.5": binm["precision"], "recall_at_0.5": binm["recall"],
            "pr_auc": binm["pr_auc"], "ece": binm["ece"],
            "crack_trigger_rate": float(np.mean(triggered[is_crack == 1])),
            "noise_triggers_per_module_day_assumption": NOISE_TRIGGERS_PER_MODULE_DAY,
            "evt_threshold": evt, "recall_at_1_far_per_module_day": recall_at_far,
            "n_noise_events_triggered": int(len(noise_scores)),
        },
        "open_set": open_set,
        "deep_model": deep,
        "localization_tier0_heldout": {**loc, "by_module": loc_by_module},
        "localization_tier1": t1,
        "targets": TARGETS,
        "target_status": {
            "recall_at_1_far": recall_at_far >= TARGETS["recall_at_1_far"],
            "macro_f1": mc["macro_f1"] >= TARGETS["macro_f1"],
            "loc_median_mm": loc["median_error_mm"] <= TARGETS["loc_median_mm"],
            "loc_p90_mm": loc["p90_error_mm"] <= TARGETS["loc_p90_mm"],
            "cell_id_accuracy": loc["cell_id_accuracy"] >= TARGETS["cell_id_accuracy"],
            "ece": binm["ece"] <= TARGETS["ece"],
        },
        "caveat": "Simulation-only (Tier-0 synthetic data). No real PV AE data exist yet; these numbers are "
                  "not field performance claims (see docs/research/CODEBASE_EVALUATION.md §6).",
    }
    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (root / "validation" / "kpi_benchmark_summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                                                    encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("feature_model", "new_fracture_detection", "localization_tier0_heldout",
                                               "localization_tier1", "open_set", "deep_model", "target_status")},
                     indent=1, default=float)[:6000])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-module", type=int, default=1200)
    args = ap.parse_args()
    main(n_per_module=args.n_per_module)
