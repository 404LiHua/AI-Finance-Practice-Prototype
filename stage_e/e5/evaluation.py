from __future__ import annotations

import math
import re
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = [
    "model_id", "seed", "fold_id", "sample_row_id", "trade_date", "target_date",
    "stock_code", "target_return", "prediction", "sample_valid", "text_available",
    "checkpoint_sha256", "config_sha256",
]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_prediction_contract(
    predictions: pd.DataFrame,
    expected_keys: pd.DataFrame,
    expected_folds: list[str],
    expected_seeds: list[int],
    require_all_frozen_keys: bool = True,
    allow_extra_rows: bool = False,
    target_tolerance: float = 1e-7,
) -> dict[str, Any]:
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise KeyError(f"E-5 predictions missing required columns: {missing}")
    if predictions.empty:
        raise ValueError("E-5 predictions cannot be empty")
    key_columns = ["model_id", "seed", "fold_id", "sample_row_id"]
    if predictions.duplicated(key_columns).any():
        raise ValueError("duplicate E-5 model/seed/fold/sample_row_id rows")
    numeric = predictions[["target_return", "prediction"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("E-5 predictions contain non-finite target or prediction")
    for column in ("checkpoint_sha256", "config_sha256"):
        if not predictions[column].astype(str).map(lambda value: bool(SHA256_PATTERN.fullmatch(value))).all():
            raise ValueError(f"invalid SHA-256 values in {column}")
    if set(predictions["fold_id"].astype(str)) != set(expected_folds):
        raise ValueError("E-5 prediction fold set differs from frozen folds")
    if set(predictions["seed"].astype(int)) != set(expected_seeds):
        raise ValueError("E-5 prediction seed set differs from frozen seeds")

    expected = expected_keys[[
        "fold_id", "sample_row_id", "trade_date", "target_date", "stock_code",
        "target_return", "sample_valid", "text_available",
    ]].copy()
    expected["fold_id"] = expected["fold_id"].astype(str)
    expected["sample_row_id"] = expected["sample_row_id"].astype(str)
    expected_key_set = set(zip(expected["fold_id"], expected["sample_row_id"]))
    rows = []
    for (model_id, seed), frame in predictions.groupby(["model_id", "seed"], sort=True):
        actual_key_set = set(zip(frame["fold_id"].astype(str), frame["sample_row_id"].astype(str)))
        missing_keys = expected_key_set - actual_key_set
        extra_keys = actual_key_set - expected_key_set
        if require_all_frozen_keys and missing_keys:
            raise ValueError(f"{model_id} seed {seed} misses {len(missing_keys)} frozen keys")
        if not allow_extra_rows and extra_keys:
            raise ValueError(f"{model_id} seed {seed} adds {len(extra_keys)} non-frozen keys")
        merged = frame.merge(
            expected, on=["fold_id", "sample_row_id"], how="inner", suffixes=("", "_expected"), validate="one_to_one"
        )
        for column in ("trade_date", "target_date", "stock_code"):
            if not merged[column].astype(str).eq(merged[f"{column}_expected"].astype(str)).all():
                raise ValueError(f"{model_id} seed {seed} changes frozen {column}")
        if not merged["sample_valid"].astype(bool).eq(merged["sample_valid_expected"].astype(bool)).all():
            raise ValueError(f"{model_id} seed {seed} changes sample_valid")
        maximum_target_difference = float(np.abs(merged["target_return"] - merged["target_return_expected"]).max())
        if maximum_target_difference > target_tolerance:
            raise ValueError(f"{model_id} seed {seed} changes target values")
        rows.append({
            "model_id": str(model_id), "seed": int(seed), "row_count": len(frame),
            "frozen_key_count": len(actual_key_set), "missing_key_count": len(missing_keys),
            "extra_key_count": len(extra_keys), "maximum_target_difference": maximum_target_difference,
        })
    return {"models": sorted(predictions["model_id"].astype(str).unique()), "grid_rows": rows}


def binary_metrics(target: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, float]:
    actual = target >= threshold
    predicted = prediction >= threshold
    tp = int(np.logical_and(actual, predicted).sum())
    tn = int(np.logical_and(~actual, ~predicted).sum())
    fp = int(np.logical_and(~actual, predicted).sum())
    fn = int(np.logical_and(actual, ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision, "recall": recall, "f1": f1, "mcc": mcc,
    }


def metric_row(frame: pd.DataFrame, mape_floor: float, threshold: float) -> dict[str, float | int]:
    target = frame["target_return"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    error = prediction - target
    classification = binary_metrics(target, prediction, threshold)
    return {
        "samples": int(len(frame)), "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(np.abs(error) / np.maximum(np.abs(target), mape_floor))),
        **classification,
    }


def grouped_metric_frame(
    frame: pd.DataFrame, group_columns: list[str], mape_floor: float, threshold: float,
) -> pd.DataFrame:
    rows = []
    grouper: Any = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in frame.groupby(grouper, dropna=False, sort=True):
        values = (key,) if len(group_columns) == 1 else key
        rows.append({**dict(zip(group_columns, values)), **metric_row(group, mape_floor, threshold)})
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame,
    universe: pd.DataFrame,
    mape_floor: float,
    direction_threshold: float,
    return_group_count: int,
) -> dict[str, pd.DataFrame]:
    valid = predictions.loc[predictions["sample_valid"].astype(bool)].copy()
    fold_metrics = grouped_metric_frame(valid, ["model_id", "seed", "fold_id"], mape_floor, direction_threshold)
    seed_metrics = grouped_metric_frame(valid, ["model_id", "seed"], mape_floor, direction_threshold)
    overall_metrics = grouped_metric_frame(valid, ["model_id"], mape_floor, direction_threshold)
    worst_fold = fold_metrics.groupby("model_id", as_index=False).agg(
        worst_fold_mae=("mae", "max"), worst_fold_rmse=("rmse", "max"), fold_count=("fold_id", "nunique")
    )
    overall_metrics = overall_metrics.merge(worst_fold, on="model_id", validate="one_to_one")
    seed_summary = seed_metrics.groupby("model_id", as_index=False).agg(
        seed_count=("seed", "nunique"), seed_mae_mean=("mae", "mean"), seed_mae_std=("mae", "std"),
        seed_rmse_mean=("rmse", "mean"), seed_accuracy_mean=("accuracy", "mean"), seed_mcc_mean=("mcc", "mean"),
    )
    seed_summary["seed_mae_std"] = seed_summary["seed_mae_std"].fillna(0.0)
    seed_summary["seed_mae_cv"] = seed_summary["seed_mae_std"] / seed_summary["seed_mae_mean"]

    pair_rows = []
    for model_id, model in valid.groupby("model_id", sort=True):
        seeds = sorted(int(seed) for seed in model["seed"].unique())
        pivot = model.pivot(index=["fold_id", "sample_row_id"], columns="seed", values="prediction")
        for left, right in combinations(seeds, 2):
            joined = pivot[[left, right]].dropna()
            left_values = joined[left].to_numpy(dtype=float)
            right_values = joined[right].to_numpy(dtype=float)
            left_constant = bool(len(left_values) and np.allclose(left_values, left_values[0], rtol=0.0, atol=1e-12))
            right_constant = bool(len(right_values) and np.allclose(right_values, right_values[0], rtol=0.0, atol=1e-12))
            if left_constant or right_constant:
                identical = bool(np.allclose(left_values, right_values, rtol=0.0, atol=1e-12))
                pearson = 1.0 if identical else 0.0
                spearman = 1.0 if identical else 0.0
            else:
                pearson = float(joined[left].corr(joined[right], method="pearson"))
                spearman = float(joined[left].corr(joined[right], method="spearman"))
            pair_rows.append({
                "model_id": model_id, "seed_a": left, "seed_b": right, "sample_count": len(joined),
                "prediction_pearson": pearson,
                "prediction_spearman": spearman,
                "prediction_sign_agreement": float(((joined[left] >= direction_threshold) == (joined[right] >= direction_threshold)).mean()),
            })
    pairwise = pd.DataFrame(pair_rows, columns=[
        "model_id", "seed_a", "seed_b", "sample_count", "prediction_pearson",
        "prediction_spearman", "prediction_sign_agreement",
    ])

    joined = valid.merge(
        universe[["stock_code", "industry_group", "market_cap_bucket_cutoff"]],
        on="stock_code", how="left", validate="many_to_one",
    )
    joined["return_decile"] = joined.groupby(["model_id", "seed"])["target_return"].transform(
        lambda values: pd.qcut(values.rank(method="first"), return_group_count, labels=[f"D{i}" for i in range(1, return_group_count + 1)])
    )
    return {
        "fold_metrics": fold_metrics,
        "seed_metrics": seed_metrics,
        "overall_metrics": overall_metrics,
        "seed_summary": seed_summary,
        "pairwise_seed_stability": pairwise,
        "diagnostics_per_stock": grouped_metric_frame(joined, ["model_id", "stock_code"], mape_floor, direction_threshold),
        "diagnostics_industry": grouped_metric_frame(joined, ["model_id", "industry_group"], mape_floor, direction_threshold),
        "diagnostics_market_cap": grouped_metric_frame(joined, ["model_id", "market_cap_bucket_cutoff"], mape_floor, direction_threshold),
        "diagnostics_return_decile": grouped_metric_frame(joined, ["model_id", "return_decile"], mape_floor, direction_threshold),
        "diagnostics_text_availability": grouped_metric_frame(joined, ["model_id", "text_available"], mape_floor, direction_threshold),
    }
