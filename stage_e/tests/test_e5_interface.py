from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stage_e.e5.evaluation import evaluate_predictions, validate_prediction_contract
from stage_e.e5.interface import E5FoldView, validation_key_frame


class E5InterfaceTest(unittest.TestCase):
    def test_validation_key_frame_preserves_row_ids(self) -> None:
        view = E5FoldView(
            fold_id="E_RO_01", numeric_values=np.zeros((1, 2, 2, 1), dtype=np.float32),
            target_raw=np.array([[0.1, -0.1]], dtype=np.float32), target_scaled=np.zeros((1, 2), dtype=np.float32),
            sample_mask=np.ones((1, 2), dtype=bool), node_available=np.ones((1, 2), dtype=bool),
            split=np.array(["validation"]), trade_date=np.array(["2023-01-01"]),
            target_date=np.array([["2023-01-08", "2023-01-08"]]), sample_row_id=np.array([["r1", "r2"]]),
            stock_code=np.array(["a", "b"]), text_features=np.zeros((1, 2, 0), dtype=np.float32),
            text_available=np.zeros((1, 2), dtype=bool), text_count=np.zeros((1, 2), dtype=np.int64),
            target_mean_train=0.0, target_std_train=1.0,
        )
        result = validation_key_frame(view)
        self.assertEqual(result["sample_row_id"].tolist(), ["r1", "r2"])

    def test_contract_and_evaluator(self) -> None:
        expected_rows = []
        prediction_rows = []
        for fold_id in ("E_RO_01", "E_RO_02", "E_RO_03"):
            for stock, target in (("a", 0.1), ("b", -0.1)):
                row_id = f"{fold_id}-{stock}"
                expected_rows.append({
                    "fold_id": fold_id, "sample_row_id": row_id, "trade_date": "2023-01-01",
                    "target_date": "2023-01-08", "stock_code": stock, "target_return": target,
                    "sample_valid": True, "text_available": False,
                })
                for seed in (1, 2, 3):
                    prediction_rows.append({
                        "model_id": "model", "seed": seed, "fold_id": fold_id, "sample_row_id": row_id,
                        "trade_date": "2023-01-01", "target_date": "2023-01-08", "stock_code": stock,
                        "target_return": target, "prediction": target * 0.8, "sample_valid": True,
                        "text_available": False, "checkpoint_sha256": "a" * 64, "config_sha256": "b" * 64,
                    })
        predictions = pd.DataFrame(prediction_rows)
        expected = pd.DataFrame(expected_rows)
        receipt = validate_prediction_contract(
            predictions, expected, ["E_RO_01", "E_RO_02", "E_RO_03"], [1, 2, 3]
        )
        self.assertEqual(receipt["models"], ["model"])
        universe = pd.DataFrame({
            "stock_code": ["a", "b"], "industry_group": ["x", "y"],
            "market_cap_bucket_cutoff": ["large", "small"],
        })
        evaluated = evaluate_predictions(predictions, universe, 1e-6, 0.0, 2)
        self.assertIn("fold_metrics", evaluated)
        self.assertAlmostEqual(float(evaluated["overall_metrics"].iloc[0]["mae"]), 0.02)

    def test_single_seed_evaluation_has_finite_summary_and_stable_empty_pair_schema(self) -> None:
        expected = pd.DataFrame([{
            "fold_id": "E_RO_01", "sample_row_id": "r1", "trade_date": "2023-01-01",
            "target_date": "2023-01-08", "stock_code": "a", "target_return": 0.1,
            "sample_valid": True, "text_available": False,
        }])
        predictions = expected.copy()
        predictions.insert(0, "seed", 1)
        predictions.insert(0, "model_id", "model")
        predictions["prediction"] = 0.08
        predictions["checkpoint_sha256"] = "a" * 64
        predictions["config_sha256"] = "b" * 64
        universe = pd.DataFrame({
            "stock_code": ["a"], "industry_group": ["x"], "market_cap_bucket_cutoff": ["large"],
        })
        evaluated = evaluate_predictions(predictions, universe, 1e-6, 0.0, 1)
        summary = evaluated["seed_summary"].select_dtypes(include=[np.number]).to_numpy()
        self.assertTrue(np.isfinite(summary).all())
        self.assertEqual(evaluated["pairwise_seed_stability"].columns.tolist(), [
            "model_id", "seed_a", "seed_b", "sample_count", "prediction_pearson",
            "prediction_spearman", "prediction_sign_agreement",
        ])

    def test_identical_constant_predictions_have_unit_seed_correlation(self) -> None:
        rows = []
        for seed in (1, 2, 3):
            for index, target in enumerate((0.1, -0.1)):
                rows.append({
                    "model_id": "deterministic", "seed": seed, "fold_id": "E_RO_01",
                    "sample_row_id": f"r{index}", "trade_date": "2023-01-01",
                    "target_date": "2023-01-08", "stock_code": str(index),
                    "target_return": target, "prediction": 0.0, "sample_valid": True,
                    "text_available": False, "checkpoint_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                })
        universe = pd.DataFrame({
            "stock_code": ["0", "1"], "industry_group": ["x", "x"],
            "market_cap_bucket_cutoff": ["large", "small"],
        })
        evaluated = evaluate_predictions(pd.DataFrame(rows), universe, 1e-6, 0.0, 2)
        pairwise = evaluated["pairwise_seed_stability"]
        self.assertTrue((pairwise["prediction_pearson"] == 1.0).all())
        self.assertTrue((pairwise["prediction_spearman"] == 1.0).all())


if __name__ == "__main__":
    unittest.main()
