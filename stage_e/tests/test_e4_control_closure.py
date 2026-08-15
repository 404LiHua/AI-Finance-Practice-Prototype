from __future__ import annotations

import unittest

import pandas as pd

from stage_e.run_e4_control_closure import ensemble_predictions, grouped_metrics


class E4ControlClosureTest(unittest.TestCase):
    def test_ensemble_keeps_frozen_key(self) -> None:
        rows = []
        for seed, prediction in ((1, 0.1), (2, 0.3)):
            rows.append({"variant": "v", "fold_id": "f", "trade_date": "2023-01-01", "stock_code": "s", "target_return": 0.2, "sample_valid": True, "text_available": False, "prediction": prediction, "seed": seed})
        result = ensemble_predictions(pd.DataFrame(rows))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result.iloc[0]["prediction"]), 0.2)

    def test_grouped_metrics_are_finite(self) -> None:
        frame = pd.DataFrame({
            "variant": ["v", "v"], "stock_code": ["a", "a"],
            "prediction": [0.1, -0.2], "target_return": [0.2, -0.1], "prediction_std": [0.01, 0.02],
        })
        result = grouped_metrics(frame, ["variant", "stock_code"])
        self.assertAlmostEqual(float(result.iloc[0]["mae"]), 0.1)


if __name__ == "__main__":
    unittest.main()
