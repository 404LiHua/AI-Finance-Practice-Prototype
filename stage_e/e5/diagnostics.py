from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from stage_e.e5.evaluation import grouped_metric_frame, metric_row


def engineering_cost_summary(receipts: pd.DataFrame) -> pd.DataFrame:
    frame = receipts.copy()
    numeric_columns = [
        "duration_seconds", "training_seconds", "inference_seconds", "epochs_completed",
        "parameter_count", "independent_load_max_abs_difference",
    ]
    for column in numeric_columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows = []
    for model_id, group in frame.groupby("model_id", sort=True):
        rows.append({
            "model_id": str(model_id), "run_count": int(len(group)),
            "fold_count": int(group["fold_id"].nunique()), "seed_count": int(group["seed"].nunique()),
            "total_duration_seconds": float(group["duration_seconds"].fillna(0.0).sum()),
            "total_training_seconds": float(group["training_seconds"].fillna(0.0).sum()),
            "recorded_inference_seconds": float(group["inference_seconds"].fillna(0.0).sum()),
            "inference_receipt_count": int(group["inference_seconds"].notna().sum()),
            "total_epochs_completed": int(group["epochs_completed"].fillna(0.0).sum()),
            "median_parameter_count": float(group["parameter_count"].dropna().median()),
            "maximum_parameter_count": float(group["parameter_count"].dropna().max()),
            "independent_load_max_abs_difference": float(group["independent_load_max_abs_difference"].max()),
        })
    return pd.DataFrame(rows)


def fold_and_worst_fold(
    predictions: pd.DataFrame, mape_floor: float, direction_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = predictions.loc[predictions["sample_valid"].astype(bool)].copy()
    fold_metrics = grouped_metric_frame(valid, ["model_id", "fold_id"], mape_floor, direction_threshold)
    worst_rows = []
    for model_id, group in fold_metrics.groupby("model_id", sort=True):
        worst = group.sort_values(["mae", "rmse", "fold_id"], ascending=[False, False, True]).iloc[0]
        worst_rows.append({
            "model_id": str(model_id), "worst_fold_id": str(worst["fold_id"]),
            "worst_fold_mae": float(worst["mae"]), "worst_fold_rmse": float(worst["rmse"]),
            "worst_fold_mse": float(worst["mse"]), "worst_fold_accuracy": float(worst["accuracy"]),
            "fold_count": int(len(group)),
        })
    return fold_metrics, pd.DataFrame(worst_rows)


def seed_prediction_dispersion(predictions: pd.DataFrame) -> pd.DataFrame:
    valid = predictions.loc[predictions["sample_valid"].astype(bool)].copy()
    dispersion = valid.groupby(["model_id", "fold_id", "sample_row_id"], sort=True)["prediction"].std(ddof=0)
    rows = []
    for model_id, values in dispersion.groupby(level=0, sort=True):
        array = values.to_numpy(dtype=float)
        rows.append({
            "model_id": str(model_id), "sample_count": int(len(array)),
            "prediction_seed_std_mean": float(np.mean(array)),
            "prediction_seed_std_median": float(np.median(array)),
            "prediction_seed_std_p95": float(np.quantile(array, 0.95)),
            "prediction_seed_std_max": float(np.max(array)),
        })
    return pd.DataFrame(rows)


def _safe_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    left_constant = bool(np.allclose(left, left[0], rtol=0.0, atol=1e-12))
    right_constant = bool(np.allclose(right, right[0], rtol=0.0, atol=1e-12))
    if left_constant or right_constant:
        return 1.0 if np.allclose(left, right, rtol=0.0, atol=1e-12) else 0.0
    series_left, series_right = pd.Series(left), pd.Series(right)
    value = float(series_left.corr(series_right, method=method))
    return value if np.isfinite(value) else 0.0


def model_disagreement(
    predictions: pd.DataFrame, component_map: dict[str, str], direction_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = predictions.loc[predictions["sample_valid"].astype(bool)].copy()
    aggregated = valid.groupby(
        ["model_id", "fold_id", "sample_row_id"], as_index=False, sort=True,
    )["prediction"].mean()
    pivot = aggregated.pivot(index=["fold_id", "sample_row_id"], columns="model_id", values="prediction")
    rows = []
    for left_model, right_model in combinations(sorted(component_map), 2):
        joined = pivot[[left_model, right_model]].dropna()
        left = joined[left_model].to_numpy(dtype=float)
        right = joined[right_model].to_numpy(dtype=float)
        difference = left - right
        rows.append({
            "model_a": left_model, "model_b": right_model,
            "component_a": component_map[left_model], "component_b": component_map[right_model],
            "sample_count": int(len(joined)),
            "prediction_pearson": _safe_correlation(left, right, "pearson"),
            "prediction_spearman": _safe_correlation(left, right, "spearman"),
            "prediction_sign_agreement": float(((left >= direction_threshold) == (right >= direction_threshold)).mean()),
            "prediction_mean_absolute_difference": float(np.mean(np.abs(difference))),
            "prediction_rmse_difference": float(np.sqrt(np.mean(difference ** 2))),
        })
    pairwise = pd.DataFrame(rows)
    pairwise["component_pair"] = pairwise.apply(
        lambda row: " | ".join(sorted([str(row["component_a"]), str(row["component_b"])])), axis=1,
    )
    component = pairwise.groupby("component_pair", as_index=False, sort=True).agg(
        model_pair_count=("model_a", "size"),
        pearson_mean=("prediction_pearson", "mean"), pearson_min=("prediction_pearson", "min"),
        spearman_mean=("prediction_spearman", "mean"), sign_agreement_mean=("prediction_sign_agreement", "mean"),
        mean_absolute_difference_mean=("prediction_mean_absolute_difference", "mean"),
        rmse_difference_mean=("prediction_rmse_difference", "mean"),
    )
    return pairwise, component


def overall_pooled_metrics(predictions: pd.DataFrame, mape_floor: float, direction_threshold: float) -> pd.DataFrame:
    valid = predictions.loc[predictions["sample_valid"].astype(bool)].copy()
    rows = []
    for model_id, group in valid.groupby("model_id", sort=True):
        rows.append({"model_id": str(model_id), **metric_row(group, mape_floor, direction_threshold)})
    return pd.DataFrame(rows)
