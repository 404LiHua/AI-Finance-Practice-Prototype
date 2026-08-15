import unittest

import pandas as pd

from stage_c.run_v2_diagnostics import quantile_group, regression_metrics


class V2DiagnosticsTest(unittest.TestCase):
    def test_regression_metrics(self) -> None:
        frame = pd.DataFrame({
            "target_return": [0.1, -0.1],
            "prediction": [0.05, -0.05],
        })
        metrics = regression_metrics(frame, "prediction")
        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["mae"], 0.05)
        self.assertAlmostEqual(metrics["rmse"], 0.05)
        self.assertEqual(metrics["direction_accuracy"], 1.0)

    def test_quantile_group_preserves_requested_bucket_count(self) -> None:
        values = pd.Series(range(20), dtype=float)
        grouped = quantile_group(values, 4, ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(set(grouped.astype(str)), {"Q1", "Q2", "Q3", "Q4"})
        self.assertTrue((grouped.value_counts() == 5).all())


if __name__ == "__main__":
    unittest.main()

