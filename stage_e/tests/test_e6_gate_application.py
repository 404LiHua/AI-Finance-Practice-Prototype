from __future__ import annotations

import unittest

import pandas as pd

from stage_e.e6.gates import apply_candidate_gates


class E6GateApplicationTest(unittest.TestCase):
    def test_all_pass_candidate_becomes_unique_without_ranking(self) -> None:
        config = {
            "baseline_roles": {"naive": "n", "strong_frozen_baseline": "f"},
            "candidate_model_ids": ["c"],
            "gates": {
                "overall": {"resolved_mae_max": 1.0, "resolved_rmse_max": 1.0},
                "worst_fold": {"required_fold_count": 3, "resolved_mae_max": 1.0},
                "per_stock": {"required_stock_count": 1, "minimum_stocks_with_mae_below_naive": 1, "resolved_maximum_stock_mae": 1.0},
                "industry": {"required_industry_count": 1, "minimum_industries_with_mae_not_above_naive": 1, "all_industry_mae_max_ratio_vs_each_baseline": 1.05, "resolved_information_technology_mae_max": 1.0},
                "market_cap": {"required_groups": ["mid"], "minimum_groups_with_mae_not_above_naive": 1, "all_group_mae_max_ratio_vs_each_baseline": 1.05, "resolved_mid_cap_mae_max": 1.0},
                "return_tails": {"required_groups": ["D1", "D10"], "resolved_d1_mae_max": 1.0, "resolved_d10_mae_max": 1.0, "resolved_tail_mean_mae_max": 1.0},
                "seed_stability": {"required_seed_count": 3, "seed_mae_cv_max": 1.0, "minimum_all_pairwise_prediction_pearson": 0.0, "minimum_all_pairwise_prediction_spearman": 0.0, "prediction_seed_std_mean_max": 1.0, "prediction_seed_std_p95_max": 1.0},
                "engineering_cost": {"required_run_receipts": 9, "total_training_seconds_max": 100.0, "total_duration_seconds_max": 100.0, "maximum_parameter_count": 100.0, "independent_load_max_abs_difference": 1e-7, "required_inference_receipt_count": 9, "recorded_inference_seconds_max": 1.0},
            },
            "eligibility_logic": {"tie_break_priority": []},
        }
        tables = {
            "overall": pd.DataFrame({"model_id": ["c"], "mae": [0.1], "rmse": [0.1]}),
            "worst_fold": pd.DataFrame({"model_id": ["c"], "worst_fold_mae": [0.1], "fold_count": [3]}),
            "per_stock": pd.DataFrame({"model_id": ["n", "c"], "stock_code": ["s", "s"], "mae": [0.2, 0.1]}),
            "industry": pd.DataFrame({"model_id": ["n", "f", "c"], "industry_group": ["信息技术"] * 3, "mae": [0.2, 0.2, 0.1]}),
            "market_cap": pd.DataFrame({"model_id": ["n", "f", "c"], "market_cap_bucket_cutoff": ["mid"] * 3, "mae": [0.2, 0.2, 0.1]}),
            "return_decile": pd.DataFrame({"model_id": ["c", "c"], "return_decile": ["D1", "D10"], "mae": [0.1, 0.1]}),
            "seed_summary": pd.DataFrame({"model_id": ["c"], "seed_count": [3], "seed_mae_cv": [0.0]}),
            "pairwise_seed": pd.DataFrame({"model_id": ["c"] * 3, "prediction_pearson": [1.0] * 3, "prediction_spearman": [1.0] * 3}),
            "seed_dispersion": pd.DataFrame({"model_id": ["c"], "prediction_seed_std_mean": [0.0], "prediction_seed_std_p95": [0.0]}),
            "cost": pd.DataFrame({"model_id": ["c"], "run_count": [9], "total_training_seconds": [1.0], "total_duration_seconds": [1.0], "maximum_parameter_count": [1], "independent_load_max_abs_difference": [0.0], "inference_receipt_count": [9], "recorded_inference_seconds": [0.1]}),
        }
        matrix, failures, outcome = apply_candidate_gates(config, tables, True)
        self.assertTrue(matrix.iloc[0]["eligible"])
        self.assertEqual(failures["c"], [])
        self.assertEqual(outcome["unique_candidate"], "c")


if __name__ == "__main__":
    unittest.main()
