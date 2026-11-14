import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from acoustisolar_schemas.module import ModuleTwinConfig, LayerConfig, ModuleLayoutConfig
from .generator import DatasetGenerator
from .baseline_model import AEBaselineGradientBooster
from .metrics import compute_classification_metrics

class HeldOutEvaluator:
    """
    Evaluator enforcing strict physical configuration splitting to prevent data leakage (PRD 9.5).
    Trains models on one set of module geometries / materials and evaluates on held-out unseen modules.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def create_held_out_configs(self) -> Tuple[ModuleTwinConfig, ModuleTwinConfig]:
        """Create standard training configuration and an unseen held-out module configuration."""
        # Config A: 120-cell standard 3.2mm glass
        cfg_train = ModuleTwinConfig(
            module_id="MOD-TRAIN-120",
            name="Standard 120 Half-Cut PERC",
            width_mm=1040.0,
            length_mm=1760.0,
        )
        
        # Config B: 144-cell large format 2.8mm glass (unseen configuration)
        cfg_held_out = ModuleTwinConfig(
            module_id="MOD-HELDOUT-144",
            name="Large Format 144 Half-Cut TOPCon",
            width_mm=1134.0,
            length_mm=2278.0,
            layout=ModuleLayoutConfig(cells=144, rows=24, cols=6, cell_size="G12"),
            stack=[
                LayerConfig(name="glass", thickness_mm=2.8, density_kg_m3=2500.0, youngs_modulus_gpa=74.0, poissons_ratio=0.21),
                LayerConfig(name="eva_front", thickness_mm=0.50, density_kg_m3=960.0, youngs_modulus_gpa=0.015, poissons_ratio=0.45),
                LayerConfig(name="si_cell", thickness_mm=0.150, density_kg_m3=2330.0, youngs_modulus_gpa=130.0, poissons_ratio=0.28),
                LayerConfig(name="eva_back", thickness_mm=0.50, density_kg_m3=960.0, youngs_modulus_gpa=0.015, poissons_ratio=0.45),
                LayerConfig(name="backsheet", thickness_mm=0.35, density_kg_m3=1200.0, youngs_modulus_gpa=3.8, poissons_ratio=0.35),
            ]
        )
        return cfg_train, cfg_held_out

    def run_held_out_benchmark(self, n_train: int = 80, n_test: int = 40) -> Dict[str, Any]:
        """
        Generate data from training module and held-out module, train baseline model,
        and evaluate generalization performance on unseen physical configuration.
        """
        cfg_train, cfg_held_out = self.create_held_out_configs()
        
        # Generate training dataset on Config A
        gen_train = DatasetGenerator(module_config=cfg_train, seed=self.seed)
        _, df_train = gen_train.generate_batch(n_events=n_train)
        
        # Generate held-out test dataset on unseen Config B
        gen_held_out = DatasetGenerator(module_config=cfg_held_out, seed=self.seed + 1000)
        _, df_held_out = gen_held_out.generate_batch(n_events=n_test)
        
        # Extract features
        X_train = np.stack(df_train["feature_vector"].values)
        y_train = df_train["is_crack"].values
        
        X_test = np.stack(df_held_out["feature_vector"].values)
        y_test = df_held_out["is_crack"].values
        
        # Train baseline model
        model = AEBaselineGradientBooster(n_estimators=20, learning_rate=0.2)
        model.fit(X_train, y_train)
        
        # Evaluate on held-out module
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        metrics["train_events"] = n_train
        metrics["test_events"] = n_test
        metrics["train_module"] = cfg_train.module_id
        metrics["held_out_module"] = cfg_held_out.module_id
        return metrics
