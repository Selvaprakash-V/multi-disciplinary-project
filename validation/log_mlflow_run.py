"""
Experiment-tracking manifest for the AcoustiSolar verification suite.

Logs **computed** results only: the scale benchmark summary, physics verification results,
V&V suite results and the reproducibility proof.  If a source file is missing the script
fails instead of substituting numbers (an earlier version logged hard-coded constants).
If MLflow is installed the metrics are also logged to the local tracking store.
"""

import json
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
SOURCES = {
    "benchmark": root / "validation" / "kpi_benchmark_summary.json",
    "physics": root / "validation" / "physics_verification_results.json",
    "vv": root / "validation" / "out" / "vv_suite_results.json",
    "reproducibility": root / "data" / "reproducibility_proof.json",
}


def _load(name: str) -> dict:
    path = SOURCES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run the corresponding validation script first")
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest() -> dict:
    bench, phys, vv, repro = (_load(k) for k in ("benchmark", "physics", "vv", "reproducibility"))
    fm, nf, loc = bench["feature_model"], bench["new_fracture_detection"], bench["localization_tier0_heldout"]
    metrics = {
        "macro_f1_feature_model": fm["macro_f1"],
        "macro_f1_ci_low": fm["macro_f1_95ci"][0],
        "macro_f1_ci_high": fm["macro_f1_95ci"][1],
        "ece_top_label": fm["ece_top_label"],
        "conformal_coverage": fm["conformal_coverage"],
        "new_fracture_pr_auc": nf["pr_auc"],
        "new_fracture_f1": nf["f1_at_0.5"],
        "recall_at_1_far_per_module_day": nf["recall_at_1_far_per_module_day"],
        "localization_median_mm": loc["median_error_mm"],
        "localization_p90_mm": loc["p90_error_mm"],
        "cell_id_accuracy": loc["cell_id_accuracy"],
        "tier1_localization_median_mm": bench["localization_tier1"]["median_err_mm"],
        "safe_laminate_s0_change_pct": phys["dispersion"]["safe_vs_free_glass"]["velocity_change_pct"],
    }
    deep = bench.get("deep_model") or {}
    if "macro_f1" in deep:
        metrics["macro_f1_cnn"] = deep["macro_f1"]
    return {
        "experiment_name": "AcoustiSolar-Verification",
        "run_id": f"run-{int(time.time())}",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": {
            "code_git_sha": repro.get("code_git_sha"),
            "config_hash": repro.get("config_hash"),
            "n_train": bench["dataset"]["n_train"],
            "n_test": bench["dataset"]["n_test"],
            "train_modules": ",".join(bench["dataset"]["train_modules"]),
            "test_modules": ",".join(bench["dataset"]["test_modules"]),
            "noise_triggers_per_module_day_assumption": nf["noise_triggers_per_module_day_assumption"],
        },
        "metrics": metrics,
        "vv_summary": vv["summary"],
        "target_status": bench["target_status"],
        "sources": {k: str(v.relative_to(root)) for k, v in SOURCES.items()},
    }


def main() -> int:
    manifest = build_manifest()
    out = root / "validation" / "out" / "experiment_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    try:
        import mlflow

        mlflow.set_experiment(manifest["experiment_name"])
        with mlflow.start_run(run_name=manifest["run_id"]):
            mlflow.log_params({k: str(v) for k, v in manifest["parameters"].items()})
            mlflow.log_metrics({k: float(v) for k, v in manifest["metrics"].items()})
            for path in SOURCES.values():
                mlflow.log_artifact(str(path))
        print("Logged to MLflow.")
    except ImportError:
        print("MLflow not installed: wrote manifest only.")
    print(f"Manifest written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
