from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage_d.custody import DataCustodyGuard
from stage_d.d2_baselines import load_locked_config, registered_models


KEY_COLUMNS = ["fold_id", "seed", "stock_code", "trade_date", "target_date"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_diagnostic_config(path: Path, repo_root: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("status") != "DIAGNOSTIC_RULES_LOCKED":
        raise ValueError("D-3 diagnostic rules must be locked")
    if config.get("candidate_additions_allowed") is not False:
        raise ValueError("D-3 cannot add candidates")
    if config.get("shrinkage_changes_allowed") is not False:
        raise ValueError("D-3 cannot change shrinkage coefficients")
    for key in ("source_root", "source_config_path", "custody_config_path", "output_root"):
        value = Path(config[key])
        config[key] = str((repo_root / value).resolve() if not value.is_absolute() else value.resolve())
    if sha256_file(Path(config["source_config_path"])) != config["source_config_sha256"]:
        raise ValueError("locked D-2 source config SHA-256 mismatch")
    return config


def assign_return_group(values: pd.Series, groups: list[dict[str, Any]]) -> pd.Series:
    result = pd.Series(index=values.index, dtype="object")
    for group in groups:
        mask = pd.Series(True, index=values.index)
        if group["minimum"] is not None:
            if group["name"] == "positive_moderate":
                mask &= values.gt(float(group["minimum"]))
            else:
                mask &= values.ge(float(group["minimum"]))
        if group["maximum"] is not None:
            maximum = float(group["maximum"])
            mask &= values.le(maximum) if group["maximum_inclusive"] else values.lt(maximum)
        result.loc[mask & result.isna()] = group["name"]
    if result.isna().any():
        raise ValueError("return-group rules do not cover every target return")
    return result


def select_unique_candidate(summary: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    gate = config["robust_gate"]
    candidates = summary.copy()
    if gate["exclude_baseline_from_recommendation"]:
        candidates = candidates[candidates["model"] != "naive"]
    candidates["passes_fold_win_gate"] = candidates["folds_beating_baseline_mae"].ge(
        int(gate["minimum_folds_beating_naive_mae"])
    )
    candidates["passes_worst_fold_gate"] = candidates[
        "worst_fold_mae_gap_vs_baseline_pct"
    ].le(float(gate["maximum_worst_fold_mae_gap_vs_naive_pct"]))
    candidates["passes_mean_improvement_gate"] = candidates[
        "mean_mae_improvement_vs_baseline_pct"
    ].gt(float(gate["minimum_mean_mae_improvement_vs_naive_pct_exclusive"]))
    candidates["passes_robust_gate"] = candidates[[
        "passes_fold_win_gate", "passes_worst_fold_gate", "passes_mean_improvement_gate",
    ]].all(axis=1)
    eligible = candidates[candidates["passes_robust_gate"]].copy()
    if eligible.empty:
        raise RuntimeError("no registered D-2 candidate passes the predefined robust gate")
    fields = [item["field"] for item in config["unique_selection_order"]]
    ascending = [bool(item["ascending"]) for item in config["unique_selection_order"]]
    eligible = eligible.sort_values(fields, ascending=ascending, kind="mergesort").reset_index(drop=True)
    eligible.insert(0, "robust_rank", np.arange(1, len(eligible) + 1))
    winner = eligible.iloc[0]
    recommendation = {
        "recommended_model": str(winner["model"]),
        "robust_rank": 1,
        "eligible_candidate_count": int(len(eligible)),
        "mae_mean": float(winner["mae_mean"]),
        "rmse_mean": float(winner["rmse_mean"]),
        "mae_cv": float(winner["mae_cv"]),
        "folds_beating_naive_mae": int(winner["folds_beating_baseline_mae"]),
        "mean_mae_improvement_vs_naive_pct": float(
            winner["mean_mae_improvement_vs_baseline_pct"]
        ),
        "worst_fold_mae_gap_vs_naive_pct": float(
            winner["worst_fold_mae_gap_vs_baseline_pct"]
        ),
        "selection_rule": config["unique_selection_order"],
        "status": "UNIQUE_DEVELOPMENT_CANDIDATE_RECOMMENDATION",
        "independent_evidence": False,
    }
    return candidates.sort_values("model"), {"eligible": eligible, "recommendation": recommendation}


def load_prediction_grid(config: dict[str, Any], repo_root: Path) -> tuple[pd.DataFrame, list[str]]:
    source_root = Path(config["source_root"])
    guard = DataCustodyGuard.from_config(Path(config["custody_config_path"]), repo_root)
    guard.assert_path_allowed(source_root, purpose="D-3 diagnostics")
    d2_config = load_locked_config(Path(config["source_config_path"]), repo_root)
    models = registered_models(d2_config)
    rows = []
    for model in models:
        base = model in d2_config["base_models"]
        parent = source_root / ("runs" if base else "derived")
        paths = sorted(parent.glob(f"*__{model}__seed*/predictions.csv"))
        if len(paths) != 9:
            raise RuntimeError(f"{model} prediction grid is incomplete: {len(paths)} != 9")
        for path in paths:
            guard.assert_path_allowed(path, purpose="D-3 diagnostics")
            parts = path.parent.name.split("__")
            if len(parts) < 3:
                raise RuntimeError("malformed D-2 prediction directory name")
            fold_id, parsed_model, seed_text = parts[0], "__".join(parts[1:-1]), parts[-1]
            if parsed_model != model:
                raise RuntimeError("prediction path model mismatch")
            frame = pd.read_csv(path, low_memory=False)
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            frame["target_date"] = pd.to_datetime(frame["target_date"])
            guard.assert_development_frame(frame)
            frame.insert(0, "seed", int(seed_text.removeprefix("seed")))
            frame.insert(0, "fold_id", fold_id)
            frame.insert(0, "model", model)
            rows.append(frame)
    predictions = pd.concat(rows, ignore_index=True)
    if len(predictions) != len(models) * 9 * 150:
        raise RuntimeError("D-3 prediction row grid is incomplete")
    if predictions.duplicated(["model", *KEY_COLUMNS]).any():
        raise RuntimeError("duplicate D-3 prediction keys")
    return predictions, models


def per_stock_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["abs_error"] = (work["prediction"] - work["target_return"]).abs()
    work["squared_error"] = (work["prediction"] - work["target_return"]) ** 2
    total_error = work.groupby("model")["abs_error"].sum().rename("model_total_abs_error")
    result = work.groupby(["model", "stock_code"], as_index=False).agg(
        samples=("target_return", "size"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
        bias=("prediction", lambda x: 0.0),
        direction_accuracy=("prediction", lambda x: 0.0),
        absolute_error_sum=("abs_error", "sum"),
    )
    # GroupBy lambdas above cannot access target values; calculate the two paired metrics separately.
    paired = work.groupby(["model", "stock_code"]).apply(
        lambda f: pd.Series({
            "bias_value": float((f["prediction"] - f["target_return"]).mean()),
            "direction_accuracy_value": float(
                ((f["prediction"] > 0) == (f["target_return"] > 0)).mean()
            ),
        }), include_groups=False,
    ).reset_index()
    result = result.drop(columns=["bias", "direction_accuracy"]).merge(
        paired, on=["model", "stock_code"], validate="one_to_one"
    ).rename(columns={"bias_value": "bias", "direction_accuracy_value": "direction_accuracy"})
    result = result.merge(total_error, on="model", validate="many_to_one")
    result["absolute_error_share_pct"] = (
        result["absolute_error_sum"] / result["model_total_abs_error"] * 100
    )
    naive = result[result["model"] == "naive"][["stock_code", "mae"]].rename(
        columns={"mae": "naive_mae"}
    )
    result = result.merge(naive, on="stock_code", validate="many_to_one")
    result["mae_improvement_vs_naive_pct"] = (
        (result["naive_mae"] - result["mae"]) / result["naive_mae"] * 100
    )
    return result.sort_values(["model", "mae"], ignore_index=True)


def return_group_diagnostics(predictions: pd.DataFrame, groups: list[dict[str, Any]]) -> pd.DataFrame:
    work = predictions.copy()
    work["return_group"] = assign_return_group(work["target_return"], groups)
    work["abs_error"] = (work["prediction"] - work["target_return"]).abs()
    work["squared_error"] = (work["prediction"] - work["target_return"]) ** 2
    result = work.groupby(["model", "return_group"], as_index=False, sort=False).agg(
        samples=("target_return", "size"),
        mean_target_return=("target_return", "mean"),
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
    )
    naive = result[result["model"] == "naive"][["return_group", "mae"]].rename(
        columns={"mae": "naive_group_mae"}
    )
    result = result.merge(naive, on="return_group", validate="many_to_one")
    result["mae_improvement_vs_naive_pct"] = (
        (result["naive_group_mae"] - result["mae"]) / result["naive_group_mae"] * 100
    )
    return result.sort_values(["model", "return_group"], ignore_index=True)


def seed_stability_diagnostics(
    metrics: pd.DataFrame, predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = metrics.groupby(["model", "seed"], as_index=False).agg(
        fold_count=("fold_id", "nunique"),
        mae_mean_across_folds=("mae", "mean"),
        mae_std_across_folds=("mae", "std"),
        rmse_mean_across_folds=("rmse", "mean"),
        direction_f1_mean_across_folds=("direction_f1", "mean"),
    )
    pivot = predictions.pivot_table(
        index=["model", "fold_id", "stock_code", "trade_date", "target_date"],
        columns="seed", values="prediction",
    )
    prediction_std = pivot.std(axis=1, ddof=1).groupby("model").agg(["mean", "median", "max"])
    prediction_std.columns = [
        "mean_cross_seed_prediction_std", "median_cross_seed_prediction_std",
        "max_cross_seed_prediction_std",
    ]
    summary = by_seed.groupby("model", as_index=False).agg(
        seed_count=("seed", "nunique"),
        seed_mae_mean=("mae_mean_across_folds", "mean"),
        seed_mae_std=("mae_mean_across_folds", "std"),
        seed_mae_min=("mae_mean_across_folds", "min"),
        seed_mae_max=("mae_mean_across_folds", "max"),
    )
    summary["seed_mae_range"] = summary["seed_mae_max"] - summary["seed_mae_min"]
    summary["seed_mae_cv"] = summary["seed_mae_std"] / summary["seed_mae_mean"]
    summary = summary.merge(prediction_std.reset_index(), on="model", validate="one_to_one")
    return by_seed.sort_values(["model", "seed"]), summary.sort_values("model")


def component_disagreement_diagnostics(
    predictions: pd.DataFrame, component_models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = predictions[predictions["model"].isin(component_models)].copy()
    pivot_prediction = base.pivot(index=KEY_COLUMNS, columns="model", values="prediction")
    pivot_target = base.pivot_table(index=KEY_COLUMNS, values="target_return", aggfunc="first")
    pair_rows = []
    fold_rows = []
    for left, right in combinations(component_models, 2):
        joined = pd.DataFrame({
            "left_prediction": pivot_prediction[left],
            "right_prediction": pivot_prediction[right],
            "target_return": pivot_target["target_return"],
        }).dropna()
        joined["absolute_disagreement"] = (
            joined["left_prediction"] - joined["right_prediction"]
        ).abs()
        joined["sign_disagreement"] = (
            (joined["left_prediction"] > 0) != (joined["right_prediction"] > 0)
        )
        joined["left_abs_error"] = (joined["left_prediction"] - joined["target_return"]).abs()
        joined["right_abs_error"] = (joined["right_prediction"] - joined["target_return"]).abs()
        pair_rows.append({
            "left_model": left,
            "right_model": right,
            "samples": len(joined),
            "prediction_correlation": float(joined["left_prediction"].corr(joined["right_prediction"])),
            "mean_absolute_disagreement": float(joined["absolute_disagreement"].mean()),
            "sign_disagreement_rate": float(joined["sign_disagreement"].mean()),
            "left_lower_error_rate": float((joined["left_abs_error"] < joined["right_abs_error"]).mean()),
            "right_lower_error_rate": float((joined["right_abs_error"] < joined["left_abs_error"]).mean()),
        })
        for fold_id, frame in joined.groupby(level="fold_id"):
            fold_rows.append({
                "left_model": left,
                "right_model": right,
                "fold_id": fold_id,
                "samples": len(frame),
                "prediction_correlation": float(frame["left_prediction"].corr(frame["right_prediction"])),
                "mean_absolute_disagreement": float(frame["absolute_disagreement"].mean()),
                "sign_disagreement_rate": float(frame["sign_disagreement"].mean()),
            })
    return pd.DataFrame(pair_rows), pd.DataFrame(fold_rows)
