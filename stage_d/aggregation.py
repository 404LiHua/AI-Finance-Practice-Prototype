from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "model", "fold_id", "seed", "samples", "mae", "rmse",
    "direction_accuracy", "direction_f1",
]


def validate_metric_grid(metrics: pd.DataFrame, baseline: str = "naive") -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in metrics.columns]
    if missing:
        raise KeyError(f"cross-fold metrics are missing columns: {missing}")
    if metrics.empty:
        raise ValueError("cross-fold metrics cannot be empty")
    if metrics.duplicated(["model", "fold_id", "seed"]).any():
        raise ValueError("duplicate model-fold-seed metric rows")
    numeric = ["samples", "mae", "rmse", "direction_accuracy", "direction_f1"]
    values = metrics[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("cross-fold metrics contain non-finite values")
    if (metrics["samples"] <= 0).any() or (metrics[["mae", "rmse"]] < 0).any().any():
        raise ValueError("sample counts must be positive and errors non-negative")
    if not metrics[["direction_accuracy", "direction_f1"]].apply(
        lambda column: column.between(0, 1)
    ).all().all():
        raise ValueError("direction metrics must be within [0, 1]")
    if baseline not in set(metrics["model"]):
        raise ValueError(f"primary baseline is missing: {baseline}")
    expected_keys = set(map(tuple, metrics[["fold_id", "seed"]].drop_duplicates().to_numpy()))
    for model, frame in metrics.groupby("model"):
        keys = set(map(tuple, frame[["fold_id", "seed"]].to_numpy()))
        if keys != expected_keys:
            raise ValueError(f"model {model} does not cover the complete fold-seed grid")


def aggregate_cross_fold(
    metrics: pd.DataFrame,
    baseline: str = "naive",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_metric_grid(metrics, baseline=baseline)
    work = metrics.copy()
    baseline_runs = work[work["model"] == baseline][["fold_id", "seed", "mae", "rmse"]].rename(
        columns={"mae": "baseline_mae", "rmse": "baseline_rmse"}
    )
    work = work.merge(baseline_runs, on=["fold_id", "seed"], how="left", validate="many_to_one")
    work["mae_improvement_vs_baseline_pct"] = (
        (work["baseline_mae"] - work["mae"]) / work["baseline_mae"] * 100
    )
    work["rmse_improvement_vs_baseline_pct"] = (
        (work["baseline_rmse"] - work["rmse"]) / work["baseline_rmse"] * 100
    )

    per_fold = work.groupby(["model", "fold_id"], as_index=False).agg(
        seeds=("seed", "nunique"),
        samples_mean=("samples", "mean"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_f1_mean=("direction_f1", "mean"),
        mae_improvement_vs_baseline_pct=("mae_improvement_vs_baseline_pct", "mean"),
        rmse_improvement_vs_baseline_pct=("rmse_improvement_vs_baseline_pct", "mean"),
    )
    baseline_folds = per_fold[per_fold["model"] == baseline][["fold_id", "mae_mean"]].rename(
        columns={"mae_mean": "baseline_fold_mae"}
    )
    per_fold = per_fold.merge(baseline_folds, on="fold_id", how="left", validate="many_to_one")
    per_fold["beats_baseline_mae"] = per_fold["mae_mean"] < per_fold["baseline_fold_mae"]
    per_fold["mae_gap_vs_baseline_pct"] = (
        (per_fold["mae_mean"] - per_fold["baseline_fold_mae"])
        / per_fold["baseline_fold_mae"] * 100
    )

    rows = []
    for model, frame in work.groupby("model", sort=True):
        fold_frame = per_fold[per_fold["model"] == model]
        mae_mean = float(frame["mae"].mean())
        mae_std = float(frame["mae"].std(ddof=1))
        rows.append({
            "model": model,
            "runs": len(frame),
            "fold_count": int(frame["fold_id"].nunique()),
            "seed_count": int(frame["seed"].nunique()),
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "mae_cv": mae_std / mae_mean if mae_mean > 0 else 0.0,
            "rmse_mean": float(frame["rmse"].mean()),
            "rmse_std": float(frame["rmse"].std(ddof=1)),
            "direction_accuracy_mean": float(frame["direction_accuracy"].mean()),
            "direction_f1_mean": float(frame["direction_f1"].mean()),
            "mean_mae_improvement_vs_baseline_pct": float(
                frame["mae_improvement_vs_baseline_pct"].mean()
            ),
            "mean_rmse_improvement_vs_baseline_pct": float(
                frame["rmse_improvement_vs_baseline_pct"].mean()
            ),
            "folds_beating_baseline_mae": int(fold_frame["beats_baseline_mae"].sum()),
            "fold_win_rate": float(fold_frame["beats_baseline_mae"].mean()),
            "worst_fold_mae": float(fold_frame["mae_mean"].max()),
            "worst_fold_mae_gap_vs_baseline_pct": float(
                fold_frame["mae_gap_vs_baseline_pct"].max()
            ),
        })
    summary = pd.DataFrame(rows).sort_values(
        ["mae_mean", "worst_fold_mae", "model"], ignore_index=True
    )
    metadata = {
        "primary_baseline": baseline,
        "models": sorted(work["model"].unique().tolist()),
        "folds": sorted(work["fold_id"].unique().tolist()),
        "seeds": sorted(int(value) for value in work["seed"].unique()),
        "metric_rows": len(work),
        "selection_evidence": "rolling-origin development only; not independent screening",
    }
    return per_fold.sort_values(["model", "fold_id"]), summary, metadata
