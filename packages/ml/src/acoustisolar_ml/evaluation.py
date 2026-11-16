"""Held-out evaluation with strict physical-configuration splits (PRD §9.5)."""

from typing import Any, Dict, Tuple

import numpy as np

from acoustisolar_schemas.event import EVENT_CLASSES
from acoustisolar_schemas.module import LayerConfig, ModuleLayoutConfig, ModuleTwinConfig

from .baseline_model import AEBaselineGradientBooster
from .generator import DatasetGenerator, feature_matrix, label_vector
from .metrics import compute_classification_metrics, compute_localization_metrics, multiclass_metrics


class HeldOutEvaluator:
    """Train on one module design, test on an unseen one.

    Tier-0 now consumes the full module configuration (stack materials, EVA properties,
    size) and sensors are placed at the corners of each module, so the held-out module
    differs physically from the training module (it did not before).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def create_held_out_configs(self) -> Tuple[ModuleTwinConfig, ModuleTwinConfig]:
        cfg_train = ModuleTwinConfig(module_id="MOD-TRAIN-120", name="Standard 120 Half-Cut PERC",
                                     width_mm=1040.0, length_mm=1760.0)
        cfg_held_out = ModuleTwinConfig(
            module_id="MOD-HELDOUT-144",
            name="Large Format 144 Half-Cut TOPCon",
            width_mm=1134.0,
            length_mm=2278.0,
            layout=ModuleLayoutConfig(cells=144, rows=24, cols=6, cell_size="G12"),
            stack=[
                LayerConfig(name="glass", thickness_mm=2.8, density_kg_m3=2500.0, youngs_modulus_gpa=74.0, poissons_ratio=0.21),
                LayerConfig(name="eva_front", thickness_mm=0.50, density_kg_m3=960.0, youngs_modulus_gpa=0.015,
                            poissons_ratio=0.45, material_model="viscoelastic_prony", damping_ratio=0.15),
                LayerConfig(name="si_cell", thickness_mm=0.150, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
                LayerConfig(name="eva_back", thickness_mm=0.50, density_kg_m3=960.0, youngs_modulus_gpa=0.015,
                            poissons_ratio=0.45, material_model="viscoelastic_prony", damping_ratio=0.15),
                LayerConfig(name="backsheet", thickness_mm=0.35, density_kg_m3=1200.0, youngs_modulus_gpa=3.8, poissons_ratio=0.35),
            ],
        )
        return cfg_train, cfg_held_out

    def run_held_out_benchmark(self, n_train: int = 80, n_test: int = 40) -> Dict[str, Any]:
        cfg_train, cfg_held_out = self.create_held_out_configs()
        _, df_train = DatasetGenerator(module_config=cfg_train, seed=self.seed).generate_batch(n_events=n_train)
        _, df_test = DatasetGenerator(module_config=cfg_held_out, seed=self.seed + 1000).generate_batch(n_events=n_test)

        X_train, X_test = feature_matrix(df_train), feature_matrix(df_test)
        yb_train, yb_test = df_train["is_crack"].values, df_test["is_crack"].values

        try:  # multi-class model when scikit-learn is available
            from .baseline_model import AEFeatureClassifier

            clf = AEFeatureClassifier(max_iter=150).fit(X_train, label_vector(df_train))
            proba = clf.predict_proba(X_test)
            p_crack = clf.new_fracture_probability(proba)
            mc = multiclass_metrics(label_vector(df_test), proba.argmax(axis=1), EVENT_CLASSES)
            model_name = "AEFeatureClassifier(HistGradientBoosting)"
        except ImportError:
            booster = AEBaselineGradientBooster(n_estimators=20, learning_rate=0.2).fit(X_train, yb_train)
            p_crack = booster.predict_proba(X_test)
            mc = None
            model_name = "AEBaselineGradientBooster"

        metrics = compute_classification_metrics(yb_test, (p_crack >= 0.5).astype(int), p_crack)
        if mc is not None:
            metrics["macro_f1_multiclass"] = mc["macro_f1"]
            metrics["multiclass"] = mc
        loc = df_test[(df_test["is_crack"] == 1) & np.isfinite(df_test["loc_x_mm"])]
        if len(loc):
            metrics["localization"] = compute_localization_metrics(
                loc[["x_mm", "y_mm"]].values, loc[["loc_x_mm", "loc_y_mm"]].values,
                list(loc["cell_id"]), list(loc["loc_cell_id"]),
            )
        metrics.update({
            "model": model_name,
            "train_events": n_train,
            "test_events": n_test,
            "train_module": cfg_train.module_id,
            "held_out_module": cfg_held_out.module_id,
        })
        return metrics
