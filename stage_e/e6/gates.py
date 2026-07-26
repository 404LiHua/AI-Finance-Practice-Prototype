from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _row(frame: pd.DataFrame, model_id: str) -> pd.Series:
    selected = frame.loc[frame["model_id"].astype(str) == model_id]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {model_id}, found {len(selected)}")
    return selected.iloc[0]


def apply_candidate_gates(
    gate_config: dict[str, Any], tables: dict[str, pd.DataFrame], source_contract_pass: bool,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    gates = gate_config["gates"]
    naive_id = gate_config["baseline_roles"]["naive"]
    frets_id = gate_config["baseline_roles"]["strong_frozen_baseline"]
    naive_stock = tables["per_stock"].loc[tables["per_stock"]["model_id"] == naive_id].set_index("stock_code")
    naive_industry = tables["industry"].loc[tables["industry"]["model_id"] == naive_id].set_index("industry_group")
    frets_industry = tables["industry"].loc[tables["industry"]["model_id"] == frets_id].set_index("industry_group")
    naive_cap = tables["market_cap"].loc[tables["market_cap"]["model_id"] == naive_id].set_index("market_cap_bucket_cutoff")
    frets_cap = tables["market_cap"].loc[tables["market_cap"]["model_id"] == frets_id].set_index("market_cap_bucket_cutoff")
    rows = []
    failures: dict[str, list[str]] = {}
    gate_columns: list[str] = []

    for model_id in gate_config["candidate_model_ids"]:
        overall = _row(tables["overall"], model_id)
        worst = _row(tables["worst_fold"], model_id)
        seed = _row(tables["seed_summary"], model_id)
        dispersion = _row(tables["seed_dispersion"], model_id)
        cost = _row(tables["cost"], model_id)
        stock = tables["per_stock"].loc[tables["per_stock"]["model_id"] == model_id].set_index("stock_code")
        industry = tables["industry"].loc[tables["industry"]["model_id"] == model_id].set_index("industry_group")
        cap = tables["market_cap"].loc[tables["market_cap"]["model_id"] == model_id].set_index("market_cap_bucket_cutoff")
        tails = tables["return_decile"].loc[
            (tables["return_decile"]["model_id"] == model_id)
            & (tables["return_decile"]["return_decile"].astype(str).isin(gates["return_tails"]["required_groups"]))
        ].set_index("return_decile")
        pairwise = tables["pairwise_seed"].loc[tables["pairwise_seed"]["model_id"] == model_id]

        shared_stocks = stock.index.intersection(naive_stock.index)
        shared_industries = industry.index.intersection(naive_industry.index).intersection(frets_industry.index)
        shared_caps = cap.index.intersection(naive_cap.index).intersection(frets_cap.index)
        checks = {
            "gate_source_contract": bool(source_contract_pass),
            "gate_overall_mae": float(overall["mae"]) <= float(gates["overall"]["resolved_mae_max"]),
            "gate_overall_rmse": float(overall["rmse"]) <= float(gates["overall"]["resolved_rmse_max"]),
            "gate_worst_fold_count": int(worst["fold_count"]) == int(gates["worst_fold"]["required_fold_count"]),
            "gate_worst_fold_mae": float(worst["worst_fold_mae"]) <= float(gates["worst_fold"]["resolved_mae_max"]),
            "gate_stock_count": len(stock) == int(gates["per_stock"]["required_stock_count"]),
            "gate_stock_below_naive_count": int((stock.loc[shared_stocks, "mae"] < naive_stock.loc[shared_stocks, "mae"]).sum()) >= int(gates["per_stock"]["minimum_stocks_with_mae_below_naive"]),
            "gate_stock_max_mae": float(stock["mae"].max()) <= float(gates["per_stock"]["resolved_maximum_stock_mae"]),
            "gate_industry_count": len(industry) == int(gates["industry"]["required_industry_count"]),
            "gate_industry_below_naive_count": int((industry.loc[shared_industries, "mae"] <= naive_industry.loc[shared_industries, "mae"]).sum()) >= int(gates["industry"]["minimum_industries_with_mae_not_above_naive"]),
            "gate_all_industries_within_baselines": bool(
                (industry.loc[shared_industries, "mae"] <= naive_industry.loc[shared_industries, "mae"] * float(gates["industry"]["all_industry_mae_max_ratio_vs_each_baseline"])).all()
                and (industry.loc[shared_industries, "mae"] <= frets_industry.loc[shared_industries, "mae"] * float(gates["industry"]["all_industry_mae_max_ratio_vs_each_baseline"])).all()
                and len(shared_industries) == len(industry)
            ),
            "gate_information_technology": "信息技术" in industry.index and float(industry.loc["信息技术", "mae"]) <= float(gates["industry"]["resolved_information_technology_mae_max"]),
            "gate_market_cap_groups": set(cap.index.astype(str)) == set(gates["market_cap"]["required_groups"]),
            "gate_market_cap_below_naive_count": int((cap.loc[shared_caps, "mae"] <= naive_cap.loc[shared_caps, "mae"]).sum()) >= int(gates["market_cap"]["minimum_groups_with_mae_not_above_naive"]),
            "gate_all_market_caps_within_baselines": bool(
                (cap.loc[shared_caps, "mae"] <= naive_cap.loc[shared_caps, "mae"] * float(gates["market_cap"]["all_group_mae_max_ratio_vs_each_baseline"])).all()
                and (cap.loc[shared_caps, "mae"] <= frets_cap.loc[shared_caps, "mae"] * float(gates["market_cap"]["all_group_mae_max_ratio_vs_each_baseline"])).all()
                and len(shared_caps) == len(cap)
            ),
            "gate_mid_cap": "mid" in cap.index and float(cap.loc["mid", "mae"]) <= float(gates["market_cap"]["resolved_mid_cap_mae_max"]),
            "gate_tail_groups": set(tails.index.astype(str)) == set(gates["return_tails"]["required_groups"]),
            "gate_d1_mae": "D1" in tails.index and float(tails.loc["D1", "mae"]) <= float(gates["return_tails"]["resolved_d1_mae_max"]),
            "gate_d10_mae": "D10" in tails.index and float(tails.loc["D10", "mae"]) <= float(gates["return_tails"]["resolved_d10_mae_max"]),
            "gate_tail_mean_mae": set(tails.index.astype(str)) == {"D1", "D10"} and float(tails.loc[["D1", "D10"], "mae"].mean()) <= float(gates["return_tails"]["resolved_tail_mean_mae_max"]),
            "gate_seed_count": int(seed["seed_count"]) == int(gates["seed_stability"]["required_seed_count"]),
            "gate_seed_mae_cv": float(seed["seed_mae_cv"]) <= float(gates["seed_stability"]["seed_mae_cv_max"]),
            "gate_seed_pair_count": len(pairwise) == 3,
            "gate_pairwise_pearson": len(pairwise) == 3 and float(pairwise["prediction_pearson"].min()) >= float(gates["seed_stability"]["minimum_all_pairwise_prediction_pearson"]),
            "gate_pairwise_spearman": len(pairwise) == 3 and float(pairwise["prediction_spearman"].min()) >= float(gates["seed_stability"]["minimum_all_pairwise_prediction_spearman"]),
            "gate_seed_std_mean": float(dispersion["prediction_seed_std_mean"]) <= float(gates["seed_stability"]["prediction_seed_std_mean_max"]),
            "gate_seed_std_p95": float(dispersion["prediction_seed_std_p95"]) <= float(gates["seed_stability"]["prediction_seed_std_p95_max"]),
            "gate_cost_receipts": int(cost["run_count"]) == int(gates["engineering_cost"]["required_run_receipts"]),
            "gate_training_seconds": float(cost["total_training_seconds"]) <= float(gates["engineering_cost"]["total_training_seconds_max"]),
            "gate_duration_seconds": float(cost["total_duration_seconds"]) <= float(gates["engineering_cost"]["total_duration_seconds_max"]),
            "gate_parameter_count": float(cost["maximum_parameter_count"]) <= float(gates["engineering_cost"]["maximum_parameter_count"]),
            "gate_independent_load": float(cost["independent_load_max_abs_difference"]) <= float(gates["engineering_cost"]["independent_load_max_abs_difference"]),
            "gate_inference_receipts": int(cost["inference_receipt_count"]) == int(gates["engineering_cost"]["required_inference_receipt_count"]),
            "gate_inference_seconds": int(cost["inference_receipt_count"]) == int(gates["engineering_cost"]["required_inference_receipt_count"]) and float(cost["recorded_inference_seconds"]) <= float(gates["engineering_cost"]["recorded_inference_seconds_max"]),
        }
        gate_columns = list(checks)
        failed = [name for name, passed in checks.items() if not bool(passed)]
        failures[model_id] = failed
        rows.append({
            "model_id": model_id,
            "overall_mae": float(overall["mae"]), "overall_rmse": float(overall["rmse"]),
            "worst_fold_mae": float(worst["worst_fold_mae"]),
            "stocks_below_naive_mae": int((stock.loc[shared_stocks, "mae"] < naive_stock.loc[shared_stocks, "mae"]).sum()),
            "maximum_stock_mae": float(stock["mae"].max()),
            "industries_below_naive_mae": int((industry.loc[shared_industries, "mae"] <= naive_industry.loc[shared_industries, "mae"]).sum()),
            "information_technology_mae": float(industry.loc["信息技术", "mae"]) if "信息技术" in industry.index else np.nan,
            "mid_cap_mae": float(cap.loc["mid", "mae"]) if "mid" in cap.index else np.nan,
            "d1_mae": float(tails.loc["D1", "mae"]) if "D1" in tails.index else np.nan,
            "d10_mae": float(tails.loc["D10", "mae"]) if "D10" in tails.index else np.nan,
            "tail_mean_mae": float(tails.loc[["D1", "D10"], "mae"].mean()) if set(tails.index.astype(str)) == {"D1", "D10"} else np.nan,
            "seed_mae_cv": float(seed["seed_mae_cv"]),
            "minimum_pairwise_pearson": float(pairwise["prediction_pearson"].min()),
            "minimum_pairwise_spearman": float(pairwise["prediction_spearman"].min()),
            "prediction_seed_std_mean": float(dispersion["prediction_seed_std_mean"]),
            "prediction_seed_std_p95": float(dispersion["prediction_seed_std_p95"]),
            "total_training_seconds": float(cost["total_training_seconds"]),
            "total_duration_seconds": float(cost["total_duration_seconds"]),
            "maximum_parameter_count": float(cost["maximum_parameter_count"]),
            "inference_receipt_count": int(cost["inference_receipt_count"]),
            "recorded_inference_seconds": float(cost["recorded_inference_seconds"]),
            "independent_load_max_abs_difference": float(cost["independent_load_max_abs_difference"]),
            **checks, "failed_gate_count": len(failed), "eligible": len(failed) == 0,
        })

    matrix = pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)
    eligible = matrix.loc[matrix["eligible"].astype(bool)].copy()
    logic = gate_config["eligibility_logic"]
    if len(eligible) == 0:
        outcome = {"status": "FORMAL_NO_PROMOTABLE_CANDIDATE", "unique_candidate": None, "eligible_models": []}
    elif len(eligible) == 1:
        outcome = {"status": "UNIQUE_CANDIDATE_RECOMMENDATION", "unique_candidate": str(eligible.iloc[0]["model_id"]), "eligible_models": [str(eligible.iloc[0]["model_id"])]}
    else:
        ordered = eligible.sort_values(
            ["worst_fold_mae", "overall_mae", "prediction_seed_std_mean", "total_training_seconds", "model_id"],
            ascending=[True, True, True, True, True],
        )
        outcome = {
            "status": "UNIQUE_CANDIDATE_BY_FROZEN_TIE_BREAK",
            "unique_candidate": str(ordered.iloc[0]["model_id"]),
            "eligible_models": sorted(eligible["model_id"].astype(str).tolist()),
            "tie_break_priority": logic["tie_break_priority"],
        }
    outcome["candidate_count"] = len(gate_config["candidate_model_ids"])
    outcome["eligible_count"] = int(matrix["eligible"].sum())
    outcome["gate_column_count"] = len(gate_columns)
    return matrix, failures, outcome
