import unittest

import pandas as pd

from stage_c.run_independent_screening import decide


class IndependentScreeningDecisionTest(unittest.TestCase):
    def candidate_rows(self) -> pd.DataFrame:
        rows = []
        for stock_index in range(30):
            for week in range(6):
                target = (week - 2.5) * 0.01
                rows.append({
                    "stock_code": f"{stock_index:06d}.SZ",
                    "trade_date": f"2024-01-{week + 1:02d}",
                    "target_date": f"2024-01-{week + 2:02d}",
                    "target_return": target,
                    "prediction": target * 0.9,
                })
        frame = pd.DataFrame(rows)
        return pd.concat([frame.assign(seed=seed) for seed in (1, 2, 3)], ignore_index=True)

    def test_pass_requires_all_pass_checks(self) -> None:
        summary = pd.DataFrame([
            {"model": "fixed_control_ensemble_v2", "mae_mean": 0.010, "mae_std": 0.0005, "rmse_mean": 0.015, "direction_accuracy_mean": 0.70, "direction_f1_mean": 0.60},
            {"model": "naive", "mae_mean": 0.020, "mae_std": 0.0, "rmse_mean": 0.021, "direction_accuracy_mean": 0.50, "direction_f1_mean": 0.0},
            {"model": "frets_return_l4", "mae_mean": 0.018, "mae_std": 0.001, "rmse_mean": 0.019, "direction_accuracy_mean": 0.55, "direction_f1_mean": 0.30},
            {"model": "minimalist_price_only_l8", "mae_mean": 0.019, "mae_std": 0.001, "rmse_mean": 0.020, "direction_accuracy_mean": 0.55, "direction_f1_mean": 0.30},
        ])
        result = decide(summary, self.candidate_rows())
        self.assertEqual(result["verdict"], "PASS")

    def test_mae_worse_than_naive_is_failure(self) -> None:
        summary = pd.DataFrame([
            {"model": "fixed_control_ensemble_v2", "mae_mean": 0.030, "mae_std": 0.001, "rmse_mean": 0.031, "direction_accuracy_mean": 0.60, "direction_f1_mean": 0.30},
            {"model": "naive", "mae_mean": 0.020, "mae_std": 0.0, "rmse_mean": 0.025, "direction_accuracy_mean": 0.50, "direction_f1_mean": 0.0},
            {"model": "frets_return_l4", "mae_mean": 0.028, "mae_std": 0.001, "rmse_mean": 0.029, "direction_accuracy_mean": 0.55, "direction_f1_mean": 0.30},
            {"model": "minimalist_price_only_l8", "mae_mean": 0.027, "mae_std": 0.001, "rmse_mean": 0.028, "direction_accuracy_mean": 0.55, "direction_f1_mean": 0.30},
        ])
        result = decide(summary, self.candidate_rows())
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(result["failure_checks"]["candidate_mae_worse_than_naive"])


if __name__ == "__main__":
    unittest.main()
