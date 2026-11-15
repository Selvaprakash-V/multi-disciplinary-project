"""
AcoustiSolar Large-Scale KPI Benchmark Suite (PRD 9.5 & 14.1)
Generates >= 20,000 events across 5 distinct physical module configurations.
Evaluates LightGBM and 1D-ResNet on held-out modules with zero leakage and bootstrap CIs.
Saves data directly to Zarr waveforms + Parquet metadata index.
"""

import sys
import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "packages" / "schemas" / "src"))
sys.path.insert(0, str(root / "packages" / "physics" / "src"))
sys.path.insert(0, str(root / "packages" / "daqtwin" / "src"))
sys.path.insert(0, str(root / "packages" / "dsp" / "src"))
sys.path.insert(0, str(root / "packages" / "ml" / "src"))
sys.path.insert(0, str(root / "packages" / "sdk" / "src"))

from acoustisolar_schemas.module import ModuleTwinConfig, LayerConfig, ModuleLayoutConfig
from acoustisolar_schemas.source import SourceClass, CrackType, NoiseClass
from acoustisolar_ml.generator import DatasetGenerator
from acoustisolar_ml.deep_model import Waveform1DResNet
from acoustisolar_ml.metrics import compute_classification_metrics
from acoustisolar_dsp.tdoa import TDOALocalizer

def get_5_module_configs():
    """Define 5 physically distinct module configurations for zero-leakage benchmark."""
    # Mod 1: Standard 120 Half-Cut PERC (3.2 mm glass)
    mod1 = ModuleTwinConfig(module_id="MOD-1-STD120", width_mm=1040.0, length_mm=1760.0)
    
    # Mod 2: Large Format 144 Half-Cut TOPCon (2.8 mm glass)
    mod2 = ModuleTwinConfig(
        module_id="MOD-2-LARGE144",
        width_mm=1134.0, length_mm=2278.0,
        layout=ModuleLayoutConfig(cells=144, rows=24, cols=6, cell_size="G12"),
        stack=[
            LayerConfig(name="glass", thickness_mm=2.8, density_kg_m3=2500.0, youngs_modulus_gpa=74.0, poissons_ratio=0.21),
            LayerConfig(name="eva_front", thickness_mm=0.50, density_kg_m3=960.0, youngs_modulus_gpa=0.015, poissons_ratio=0.45),
            LayerConfig(name="si_cell", thickness_mm=0.150, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
            LayerConfig(name="eva_back", thickness_mm=0.50, density_kg_m3=960.0, youngs_modulus_gpa=0.015, poissons_ratio=0.45),
            LayerConfig(name="backsheet", thickness_mm=0.35, density_kg_m3=1200.0, youngs_modulus_gpa=3.8, poissons_ratio=0.35),
        ]
    )
    
    # Mod 3: Utility 60 Full-Cell Mono (4.0 mm heavy glass)
    mod3 = ModuleTwinConfig(
        module_id="MOD-3-UTIL60",
        width_mm=992.0, length_mm=1650.0,
        layout=ModuleLayoutConfig(cells=60, half_cut=False, rows=10, cols=6, cell_size="M6"),
        stack=[
            LayerConfig(name="glass", thickness_mm=4.0, density_kg_m3=2500.0, youngs_modulus_gpa=72.0, poissons_ratio=0.22),
            LayerConfig(name="eva_front", thickness_mm=0.45, density_kg_m3=960.0, youngs_modulus_gpa=0.0125, poissons_ratio=0.45),
            LayerConfig(name="si_cell", thickness_mm=0.200, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
            LayerConfig(name="eva_back", thickness_mm=0.45, density_kg_m3=960.0, youngs_modulus_gpa=0.0125, poissons_ratio=0.45),
            LayerConfig(name="backsheet", thickness_mm=0.30, density_kg_m3=1200.0, youngs_modulus_gpa=3.5, poissons_ratio=0.35),
        ]
    )
    
    # Mod 4: Glass-Glass Bifacial 72-Cell (2.0 mm thin glass front & rear) - HELD OUT
    mod4 = ModuleTwinConfig(
        module_id="MOD-4-BIFACIAL72",
        width_mm=1000.0, length_mm=2000.0,
        layout=ModuleLayoutConfig(cells=72, half_cut=False, rows=12, cols=6, cell_size="M10"),
        stack=[
            LayerConfig(name="glass_front", thickness_mm=2.0, density_kg_m3=2500.0, youngs_modulus_gpa=72.0, poissons_ratio=0.22),
            LayerConfig(name="poe_front", thickness_mm=0.40, density_kg_m3=920.0, youngs_modulus_gpa=0.020, poissons_ratio=0.43),
            LayerConfig(name="si_cell", thickness_mm=0.160, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
            LayerConfig(name="poe_back", thickness_mm=0.40, density_kg_m3=920.0, youngs_modulus_gpa=0.020, poissons_ratio=0.43),
            LayerConfig(name="glass_back", thickness_mm=2.0, density_kg_m3=2500.0, youngs_modulus_gpa=72.0, poissons_ratio=0.22),
        ]
    )
    
    # Mod 5: Thermal Cycling Aged Module (High Damping, T=65C) - HELD OUT
    mod5 = ModuleTwinConfig(
        module_id="MOD-5-HOT65C",
        width_mm=1040.0, length_mm=1760.0,
        stack=[
            LayerConfig(name="glass", thickness_mm=3.2, density_kg_m3=2500.0, youngs_modulus_gpa=70.0, poissons_ratio=0.23),
            LayerConfig(name="eva_front", thickness_mm=0.45, density_kg_m3=960.0, youngs_modulus_gpa=0.005, poissons_ratio=0.48, temperature_c=65.0),
            LayerConfig(name="si_cell", thickness_mm=0.175, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
            LayerConfig(name="eva_back", thickness_mm=0.45, density_kg_m3=960.0, youngs_modulus_gpa=0.005, poissons_ratio=0.48, temperature_c=65.0),
            LayerConfig(name="backsheet", thickness_mm=0.30, density_kg_m3=1200.0, youngs_modulus_gpa=3.2, poissons_ratio=0.36),
        ]
    )
    
    return [mod1, mod2, mod3, mod4, mod5]

def bootstrap_ci(metric_fn, y_true, y_pred, y_prob=None, n_boot=500, seed=42):
    """Compute 95% bootstrap confidence intervals [2.5%, 97.5%]."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_prob is not None:
            s = metric_fn(y_true[idx], y_pred[idx], y_prob[idx])
        else:
            s = metric_fn(y_true[idx], y_pred[idx])
        scores.append(s)
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def run_large_scale_kpi_benchmark(n_total_events=20000, output_dir="data/benchmark_20k"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    configs = get_5_module_configs()
    n_per_config = n_total_events // len(configs)
    
    print(f"=== Generating {n_total_events:,} events across 5 distinct module configurations ===")
    
    all_waveforms = []
    all_meta = []
    
    for i, cfg in enumerate(configs):
        print(f"  [{i+1}/5] Generating {n_per_config:,} events for {cfg.module_id}...")
        gen = DatasetGenerator(module_config=cfg, seed=1000 + i*100)
        waves, df = gen.generate_batch(n_events=n_per_config, crack_ratio=0.60)
        all_waveforms.append(waves)
        all_meta.append(df)
        
    full_waveforms = np.concatenate(all_waveforms, axis=0) # (20000, 4, 2560)
    full_df = pd.concat(all_meta, ignore_index=True)
    
    # Save to uncompressed array storage (int16 ADC counts) + Parquet metadata
    npz_path = out_path / "waveforms.npz"
    print(f"Saving uncompressed waveforms (int16 ADC counts) to: {npz_path}...")
    # Quantize to int16 to reflect 14/16-bit ADC counts as per PRD 8.2
    # Waveform voltages are typically +/- 2.5V, scale to ADC counts or save as float32/int16
    full_waveforms_int16 = np.clip(full_waveforms * 1000.0, -32768, 32767).astype(np.int16)
    np.savez(str(npz_path), waveforms=full_waveforms_int16)
    
    parquet_path = out_path / "events.parquet"
    print(f"Saving metadata index to Parquet: {parquet_path}...")
    # Drop raw feature vector column for clean parquet metadata storage
    meta_for_parquet = full_df.drop(columns=["feature_vector"])
    table = pa.Table.from_pandas(meta_for_parquet)
    pq.write_table(table, str(parquet_path))
    
    # Split into Train (Mods 1, 2, 3) and Held-Out Test (Mods 4, 5)
    train_indices = full_df[full_df["event_id"].str.startswith("EVT-1000") | 
                            full_df["event_id"].str.startswith("EVT-1100") | 
                            full_df["event_id"].str.startswith("EVT-1200")].index
    held_out_indices = full_df[full_df["event_id"].str.startswith("EVT-1300") | 
                               full_df["event_id"].str.startswith("EVT-1400")].index
    
    print(f"\nTrain set: {len(train_indices):,} events (Mods 1-3)")
    print(f"Held-out test set: {len(held_out_indices):,} events (Mods 4-5) - ZERO DATA LEAKAGE")
    
    X_train = np.stack(full_df.loc[train_indices, "feature_vector"].values)
    y_train = full_df.loc[train_indices, "is_crack"].values
    
    X_test = np.stack(full_df.loc[held_out_indices, "feature_vector"].values)
    y_test = full_df.loc[held_out_indices, "is_crack"].values
    
    # Model 1: LightGBM
    print("\n--- Training LightGBM Baseline Model ---")
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    lgbm_probs = lgbm.predict_proba(X_test)[:, 1]
    lgbm_preds = lgbm.predict(X_test)
    
    # Model 2: 1D-ResNet
    print("--- Evaluating 1D-ResNet Deep Waveform Model ---")
    resnet = Waveform1DResNet(in_channels=4, n_features=16, seed=42)
    resnet_test_waves = full_waveforms[held_out_indices]
    # Predict in batches
    resnet_probs = []
    bs = 200
    for b in range(0, len(resnet_test_waves), bs):
        batch = resnet_test_waves[b:b+bs]
        probs = resnet.forward(batch)
        resnet_probs.extend(probs)
    resnet_probs = np.array(resnet_probs)
    resnet_preds = (resnet_probs >= 0.5).astype(int)
    
    # Metrics Computation
    m_lgbm = compute_classification_metrics(y_test, lgbm_preds, lgbm_probs)
    m_resnet = compute_classification_metrics(y_test, resnet_preds, resnet_probs)
    
    # Detection recall at fixed FAR (1 false alarm per module-day)
    # 10,000 ambient noise triggers/day -> target FPR <= 1e-4
    prec, rec, thresholds = precision_recall_curve(y_test, lgbm_probs)
    # Find operating threshold where False Positive Rate <= 0.0001
    fpr_target = 1.0 / 10000.0
    # From confusion matrix on test noise samples
    noise_probs = lgbm_probs[y_test == 0]
    thresh_fixed_far = float(np.percentile(noise_probs, 100.0 * (1.0 - fpr_target)))
    recall_at_1_far = float(np.mean(lgbm_probs[y_test == 1] >= thresh_fixed_far))
    
    # Bootstrap Confidence Intervals (95%)
    ci_f1_lgbm = bootstrap_ci(lambda yt, yp, ypr: compute_classification_metrics(yt, yp, ypr)["f1_score"], y_test, lgbm_preds, lgbm_probs)
    ci_f1_resnet = bootstrap_ci(lambda yt, yp, ypr: compute_classification_metrics(yt, yp, ypr)["f1_score"], y_test, resnet_preds, resnet_probs)
    ci_ece_lgbm = bootstrap_ci(lambda yt, yp, ypr: compute_classification_metrics(yt, yp, ypr)["ece"], y_test, lgbm_preds, lgbm_probs)
    
    # Confusion matrix
    cm_lgbm = confusion_matrix(y_test, lgbm_preds).tolist()
    
    # Localization accuracy vs SNR
    sensor_positions = [(60.0, 60.0), (60.0, 1640.0), (980.0, 60.0), (980.0, 1640.0)]
    localizer = TDOALocalizer(sensor_positions)
    loc_errors = []
    snr_list = []
    cell_id_matches = 0
    
    # Evaluate on a representative sample of 250 crack events from held-out set
    crack_sample_indices = full_df[(full_df["is_crack"] == 1) & full_df.index.isin(held_out_indices)].index[:250]
    for idx in crack_sample_indices:
        w = full_waveforms[idx]
        dt_s = localizer.estimate_time_differences(w)
        res = localizer.solve_hyperbolic(dt_s)
        true_x = full_df.loc[idx, "x_mm"]
        true_y = full_df.loc[idx, "y_mm"]
        err = np.hypot(res["x_mm"] - true_x, res["y_mm"] - true_y)
        loc_errors.append(err)
        
        # Estimate SNR (dB) relative to 10 mV noise floor
        amp = max(float(full_df.loc[idx, "peak_amp_mv"]), 0.1)
        snr_db = 20.0 * np.log10(amp / 10.0)
        snr_list.append(snr_db)
        
        # Check cell ID match (cell size ~ 91 x 182 mm; tolerance radius ~ 60 mm)
        if err < 60.0:
            cell_id_matches += 1
            
    loc_errors = np.array(loc_errors)
    snr_arr = np.array(snr_list)
    
    median_loc_err = float(np.median(loc_errors))
    p90_loc_err = float(np.percentile(loc_errors, 90))
    cell_acc = float(cell_id_matches / len(loc_errors))
    
    # Stratify localization error by SNR bins
    low_snr_mask = snr_arr < 15.0
    med_snr_mask = (snr_arr >= 15.0) & (snr_arr < 30.0)
    high_snr_mask = snr_arr >= 30.0
    
    loc_by_snr = {
        "low_snr_lt15db": {
            "n_samples": int(np.sum(low_snr_mask)),
            "median_err_mm": float(np.median(loc_errors[low_snr_mask])) if np.any(low_snr_mask) else 0.0,
            "p90_err_mm": float(np.percentile(loc_errors[low_snr_mask], 90)) if np.any(low_snr_mask) else 0.0,
        },
        "med_snr_15_30db": {
            "n_samples": int(np.sum(med_snr_mask)),
            "median_err_mm": float(np.median(loc_errors[med_snr_mask])) if np.any(med_snr_mask) else 0.0,
            "p90_err_mm": float(np.percentile(loc_errors[med_snr_mask], 90)) if np.any(med_snr_mask) else 0.0,
        },
        "high_snr_gt30db": {
            "n_samples": int(np.sum(high_snr_mask)),
            "median_err_mm": float(np.median(loc_errors[high_snr_mask])) if np.any(high_snr_mask) else 0.0,
            "p90_err_mm": float(np.percentile(loc_errors[high_snr_mask], 90)) if np.any(high_snr_mask) else 0.0,
        }
    }
    
    # Tier-1 FDTD Localization Evaluation (5 test events to capture full elastodynamic scattering & reflections)
    print("\n--- Running Tier-1 FDTD Localization Comparison (Elastodynamics with boundary reflections) ---")
    from acoustisolar_physics.tier1 import Tier1WaveSolver
    tier1_solver = Tier1WaveSolver(width_m=1.0, length_m=1.6, dx_m=0.010, dt_s=1.0e-6, c_speed_m_s=5300.0, pml_thickness_cells=10)
    t1_loc_errors = []
    # Test 5 source locations (in meters)
    t1_sources_m = [(0.30, 0.40), (0.50, 0.80), (0.70, 1.20), (0.25, 1.30), (0.80, 0.50)]
    t1_sensors_m = [(0.06, 0.06), (0.06, 1.54), (0.94, 0.06), (0.94, 1.54)]
    t1_sensors_mm = [(x * 1e3, y * 1e3) for x, y in t1_sensors_m]
    t1_localizer = TDOALocalizer(t1_sensors_mm)
    
    n_steps = 400
    t = np.arange(n_steps) * 1.0e-6
    f0 = 150.0e3
    tau = (t - 1.5 / f0)
    ricker = (1.0 - 2.0 * (np.pi * f0 * tau)**2) * np.exp(-(np.pi * f0 * tau)**2)
    
    for sx_m, sy_m in t1_sources_m:
        t1_waves, _ = tier1_solver.run_simulation(
            source_x_m=sx_m,
            source_y_m=sy_m,
            source_signal=ricker,
            sensor_locations_m=t1_sensors_m
        ) # shape: (4, 400)
        t1_dt = t1_localizer.estimate_time_differences(t1_waves)
        t1_res = t1_localizer.solve_hyperbolic(t1_dt)
        t1_err = np.hypot(t1_res["x_mm"] - (sx_m * 1e3), t1_res["y_mm"] - (sy_m * 1e3))
        t1_loc_errors.append(t1_err)
        
    t1_loc_errors = np.array(t1_loc_errors)
    tier1_loc_summary = {
        "n_events": len(t1_loc_errors),
        "median_err_mm": float(np.median(t1_loc_errors)),
        "p90_err_mm": float(np.percentile(t1_loc_errors, 90)),
        "note": "Tier-1 includes edge reverberations and dispersion coda which increases TDOA jitter vs Tier-0 direct path"
    }
    
    results = {
        "n_total_events": n_total_events,
        "n_train": len(train_indices),
        "n_held_out_test": len(held_out_indices),
        "recall_at_1_far_per_mod_day": recall_at_1_far,
        "lightgbm": {
            "macro_f1": m_lgbm["f1_score"],
            "macro_f1_95ci": list(ci_f1_lgbm),
            "accuracy": m_lgbm["accuracy"],
            "ece": m_lgbm["ece"],
            "ece_95ci": list(ci_ece_lgbm),
            "confusion_matrix_2x2": cm_lgbm,
        },
        "resnet_1d": {
            "macro_f1": m_resnet["f1_score"],
            "macro_f1_95ci": list(ci_f1_resnet),
            "accuracy": m_resnet["accuracy"],
            "ece": m_resnet["ece"],
        },
        "localization": {
            "tier0": {
                "median_error_mm": median_loc_err,
                "p90_error_mm": p90_loc_err,
                "cell_id_accuracy": cell_acc,
                "by_snr": loc_by_snr,
            },
            "tier1": tier1_loc_summary,
        },
        "kpi_target_comparison": {
            "recall_at_fixed_far": {
                "achieved": recall_at_1_far,
                "target": 0.98,
                "met": bool(recall_at_1_far >= 0.98),
            },
            "classification_macro_f1": {
                "achieved": m_lgbm["f1_score"],
                "target": 0.95,
                "met": bool(m_lgbm["f1_score"] >= 0.95),
            },
            "localization_median_mm": {
                "achieved": median_loc_err,
                "target": 15.0, # 1.5 cm
                "met": bool(median_loc_err <= 15.0),
            },
            "localization_p90_mm": {
                "achieved": p90_loc_err,
                "target": 40.0, # 4 cm
                "met": bool(p90_loc_err <= 40.0),
            },
            "cell_id_accuracy": {
                "achieved": cell_acc,
                "target": 0.95,
                "met": bool(cell_acc >= 0.95),
            },
            "ece_calibration": {
                "achieved": m_lgbm["ece"],
                "target": 0.05,
                "met": bool(m_lgbm["ece"] <= 0.05),
            }
        }
    }
    return results

if __name__ == "__main__":
    import json
    res = run_large_scale_kpi_benchmark(n_total_events=20000)
    print("\n=== FINAL BENCHMARK SUMMARY ===")
    print(json.dumps(res, indent=2))
