from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage_e.e5.interface import E5FoldView
from stage_e.e5.low_cost import flattened_samples, load_predict_industry_var, train_industry_var


REPO_ROOT = Path(__file__).resolve().parents[2]


def fixture_view() -> E5FoldView:
    values = np.arange(3 * 4 * 2 * 2, dtype=np.float32).reshape(3, 4, 2, 2)
    return E5FoldView(
        fold_id="E_RO_01", numeric_values=values,
        target_raw=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        target_scaled=np.zeros((3, 2), dtype=np.float32), sample_mask=np.ones((3, 2), dtype=bool),
        node_available=np.ones((3, 2), dtype=bool), split=np.array(["train", "train", "validation"]),
        trade_date=np.array(["d1", "d2", "d3"]),
        target_date=np.array([["t1", "t1"], ["t2", "t2"], ["t3", "t3"]]),
        sample_row_id=np.array([["r1", "r2"], ["r3", "r4"], ["r5", "r6"]]),
        stock_code=np.array(["a", "b"]), text_features=np.zeros((3, 2, 0), dtype=np.float32),
        text_available=np.zeros((3, 2), dtype=bool), text_count=np.zeros((3, 2), dtype=np.int64),
        target_mean_train=0.0, target_std_train=1.0,
    )


class E5LowCostBaselineTest(unittest.TestCase):
    def test_flattened_samples_keep_time_then_stock_order(self) -> None:
        view = fixture_view()
        x, y, mask = flattened_samples(view, "validation", 2, [0, 1], False)
        expected = view.numeric_values[2:3, -2:, :, :].transpose(0, 2, 1, 3).reshape(2, 2, 2)
        np.testing.assert_array_equal(x, expected)
        np.testing.assert_allclose(y, [0.5, 0.6])
        self.assertEqual(mask.tolist(), [True, True])

    def test_industry_var_independent_load_matches(self) -> None:
        view = fixture_view()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "var.npz"
            prediction, _ = train_industry_var(
                view, np.array([0, 1]), np.array(["x", "y"]), 0.001, checkpoint,
            )
            loaded = load_predict_industry_var(checkpoint, view)
        np.testing.assert_allclose(prediction, loaded, rtol=0.0, atol=1e-12)

    def test_locked_model_collection_and_restrictions(self) -> None:
        config = json.loads((REPO_ROOT / "stage_e/configs/e5_low_cost_baselines_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [item["id"] for item in config["models"]],
            [
                "naive", "frets_return_l4__fixed_shrink_a075", "minimalist_price_only_l8",
                "random_forest_price_l12", "svm_rbf_price_l12", "industry_var1_ridge",
            ],
        )
        self.assertEqual(config["engineering_seed"], 20260725)
        self.assertFalse(config["restrictions"]["candidate_selection_allowed"])
        self.assertFalse(config["restrictions"]["model_deletion_allowed"])
        self.assertTrue(config["independent_loading"]["required"])
        self.assertEqual(config["independent_loading"]["maximum_prediction_absolute_difference"], 1e-7)

    def test_three_seed_review_only_adds_the_two_remaining_seeds(self) -> None:
        base_path = REPO_ROOT / "stage_e/configs/e5_low_cost_baselines_v1.json"
        review = json.loads((REPO_ROOT / "stage_e/configs/e5_low_cost_three_seed_review_v1.json").read_text(encoding="utf-8"))
        import hashlib
        self.assertEqual(hashlib.sha256(base_path.read_bytes()).hexdigest(), review["base_protocol_config_sha256"])
        self.assertEqual(review["seeds"], [20260723, 20260724, 20260725])
        self.assertEqual(review["new_training_seeds"], [20260723, 20260724])
        self.assertEqual(review["reused_engineering_seed"], 20260725)
        self.assertFalse(any(review["restrictions"].values()))


if __name__ == "__main__":
    unittest.main()
