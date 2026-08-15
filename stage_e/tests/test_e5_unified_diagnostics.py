from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stage_e.e5.diagnostics import (
    engineering_cost_summary, fold_and_worst_fold, model_disagreement, seed_prediction_dispersion,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class E5UnifiedDiagnosticsTest(unittest.TestCase):
    def test_cost_summary_uses_all_receipts(self) -> None:
        receipts = pd.DataFrame({
            "model_id": ["a", "a"], "fold_id": ["f1", "f2"], "seed": [1, 1],
            "duration_seconds": [2.0, 3.0], "training_seconds": [1.0, 2.0],
            "parameter_count": [10, 10], "independent_load_max_abs_difference": [0.0, 1e-9],
        })
        result = engineering_cost_summary(receipts).iloc[0]
        self.assertEqual(result["run_count"], 2)
        self.assertEqual(result["total_duration_seconds"], 5.0)
        self.assertEqual(result["maximum_parameter_count"], 10.0)

    def test_worst_fold_is_maximum_pooled_mae(self) -> None:
        rows = []
        for fold, prediction in (("f1", 0.0), ("f2", 0.5)):
            for seed in (1, 2, 3):
                rows.append({
                    "model_id": "m", "fold_id": fold, "seed": seed,
                    "sample_row_id": f"{fold}-{seed}", "target_return": 0.0,
                    "prediction": prediction, "sample_valid": True,
                })
        _, worst = fold_and_worst_fold(pd.DataFrame(rows), 1e-6, 0.0)
        self.assertEqual(worst.iloc[0]["worst_fold_id"], "f2")

    def test_model_disagreement_has_n_choose_two_pairs(self) -> None:
        rows = []
        component_map = {f"m{i}": f"c{i}" for i in range(4)}
        for model_index, model in enumerate(component_map):
            for seed in (1, 2, 3):
                for sample in range(3):
                    rows.append({
                        "model_id": model, "fold_id": "f", "seed": seed,
                        "sample_row_id": str(sample), "prediction": model_index + sample * 0.1,
                        "sample_valid": True,
                    })
        pairwise, component = model_disagreement(pd.DataFrame(rows), component_map, 0.0)
        self.assertEqual(len(pairwise), 6)
        self.assertEqual(len(component), 6)
        self.assertEqual(len(seed_prediction_dispersion(pd.DataFrame(rows))), 4)

    def test_frozen_config_has_exact_ten_models_and_no_training(self) -> None:
        config = json.loads((REPO_ROOT / "stage_e/configs/e5_unified_diagnostics_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(len(config["models"]), 10)
        self.assertEqual(config["diagnostic_rules"]["model_pair_count"], 45)
        self.assertFalse(config["restrictions"]["training_allowed"])
        self.assertFalse(any(config["restrictions"].values()))


if __name__ == "__main__":
    unittest.main()
