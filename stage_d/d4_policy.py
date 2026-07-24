from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from experiments.core import evaluate_predictions, prediction_frame
from stage_d.d3_diagnostics import assign_return_group


def _check(value: float, operator: str, threshold: float) -> bool:
    if operator == "le":
        return value <= threshold
    if operator == "ge":
        return value >= threshold
    if operator == "lt":
        return value < threshold
    if operator == "gt":
        return value > threshold
    raise ValueError(f"unsupported rule operator: {operator}")


def evaluate_frozen_policy(
    samples: pd.DataFrame,
    candidate_prediction: np.ndarray,
    seed_predictions: dict[int, np.ndarray],
    freeze_config: dict[str, Any],
    return_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    samples = samples.reset_index(drop=True).copy()
    coverage = freeze_config["coverage_requirements"]
    if samples["stock_code"].nunique() != int(coverage["stock_count"]):
        return {"outcome": "INVALID_INTEGRITY_FAILURE", "reason": "stock_count"}
    if samples.groupby("stock_code").size().min() < int(coverage["minimum_eligible_samples_per_stock"]):
        return {"outcome": "INVALID_INTEGRITY_FAILURE", "reason": "per_stock_coverage"}
    arrays = [np.asarray(candidate_prediction, dtype=float), *map(np.asarray, seed_predictions.values())]
    if any(len(values) != len(samples) or not np.isfinite(values).all() for values in arrays):
        return {"outcome": "INVALID_INTEGRITY_FAILURE", "reason": "prediction_integrity"}

    candidate_frame = prediction_frame(samples, candidate_prediction, "evaluation")
    naive_prediction = np.zeros(len(samples), dtype=float)
    naive_frame = prediction_frame(samples, naive_prediction, "evaluation")
    candidate_metrics = evaluate_predictions(candidate_frame)["aggregate"]
    naive_metrics = evaluate_predictions(naive_frame)["aggregate"]
    seed_maes = []
    for values in seed_predictions.values():
        seed_maes.append(evaluate_predictions(prediction_frame(samples, values, "evaluation"))["aggregate"]["mae"])
    seed_mae_cv = float(np.std(seed_maes, ddof=1) / np.mean(seed_maes))

    stock_rows = []
    for stock_code, positions in samples.groupby("stock_code").groups.items():
        index = np.asarray(list(positions), dtype=int)
        truth = samples.iloc[index]["target_return"].to_numpy(float)
        candidate = candidate_prediction[index]
        naive = np.zeros(len(index))
        candidate_mae = float(np.mean(np.abs(candidate - truth)))
        naive_mae = float(np.mean(np.abs(naive - truth)))
        stock_rows.append({
            "stock_code": stock_code,
            "candidate_mae": candidate_mae,
            "naive_mae": naive_mae,
            "mae_ratio": candidate_mae / naive_mae if naive_mae > 0 else float("inf"),
            "absolute_error_sum": float(np.abs(candidate - truth).sum()),
        })
    stock = pd.DataFrame(stock_rows)
    stocks_beating = int((stock["candidate_mae"] <= stock["naive_mae"]).sum())
    stock["error_share"] = stock["absolute_error_sum"] / stock["absolute_error_sum"].sum()
    worst_five_share = float(stock.nlargest(5, "error_share")["error_share"].sum())
    worst_single_ratio = float(stock["mae_ratio"].max())

    grouped = samples[["target_return"]].copy()
    grouped["return_group"] = assign_return_group(grouped["target_return"], return_groups)
    grouped["candidate_abs_error"] = np.abs(candidate_prediction - grouped["target_return"].to_numpy(float))
    grouped["naive_abs_error"] = np.abs(grouped["target_return"].to_numpy(float))
    ratios = {}
    for name, frame in grouped.groupby("return_group"):
        naive_mae = float(frame["naive_abs_error"].mean())
        ratios[str(name)] = float(frame["candidate_abs_error"].mean() / naive_mae)

    values = {
        "candidate_mae_le_naive": candidate_metrics["mae"] / naive_metrics["mae"],
        "candidate_rmse_within_5pct_naive": candidate_metrics["rmse"] / naive_metrics["rmse"],
        "direction_accuracy_floor": candidate_metrics["direction_accuracy"],
        "direction_f1_floor": candidate_metrics["direction_f1"],
        "three_seed_mae_cv_ceiling": seed_mae_cv,
        "stocks_beating_naive_floor": stocks_beating,
        "worst_five_stock_error_share_ceiling": worst_five_share,
        "worst_single_stock_mae_ratio_ceiling": worst_single_ratio,
        "negative_tail_mae_ratio_ceiling": ratios["negative_tail"],
        "positive_tail_mae_ratio_ceiling": ratios["positive_tail"],
        "near_zero_mae_ratio_ceiling": ratios["near_zero"],
    }
    failure_values = {
        "candidate_mae_worse_than_naive": values["candidate_mae_le_naive"],
        "candidate_rmse_more_than_10pct_worse": values["candidate_rmse_within_5pct_naive"],
        "direction_accuracy_failure_floor": values["direction_accuracy_floor"],
        "direction_f1_failure_floor": values["direction_f1_floor"],
        "three_seed_mae_cv_failure_ceiling": seed_mae_cv,
        "stocks_beating_naive_failure_floor": stocks_beating,
        "worst_five_stock_error_share_failure_ceiling": worst_five_share,
        "worst_single_stock_mae_ratio_failure_ceiling": worst_single_ratio,
        "negative_tail_mae_ratio_failure_ceiling": ratios["negative_tail"],
        "positive_tail_mae_ratio_failure_ceiling": ratios["positive_tail"],
        "near_zero_mae_ratio_failure_ceiling": ratios["near_zero"],
    }
    pass_checks = [{**rule, "value": values[rule["rule"]], "passed": _check(
        values[rule["rule"]], rule["operator"], float(rule["threshold"])
    )} for rule in freeze_config["pass_all_required"]]
    failure_checks = [{**rule, "value": failure_values[rule["rule"]], "triggered": _check(
        failure_values[rule["rule"]], rule["operator"], float(rule["threshold"])
    )} for rule in freeze_config["performance_failure_if_any"]]
    outcome = "FAIL" if any(item["triggered"] for item in failure_checks) else (
        "PASS" if all(item["passed"] for item in pass_checks) else "INCONCLUSIVE"
    )
    return {
        "outcome": outcome,
        "candidate_metrics": candidate_metrics,
        "naive_metrics": naive_metrics,
        "seed_maes": seed_maes,
        "diagnostic_values": values,
        "pass_checks": pass_checks,
        "failure_checks": failure_checks,
        "per_stock": stock.to_dict(orient="records"),
        "return_group_mae_ratios": ratios,
    }
