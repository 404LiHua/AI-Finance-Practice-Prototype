"""Run frozen E-4 fallback controls, stability review, and grouped diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve
from stage_e.run_e4_ablations import load_fold, load_graphs, run_fold
from stage_e.run_e4_fixed_graph_stabilization import prediction_correlation


FOLD_IDS = ("E_RO_01", "E_RO_02", "E_RO_03")


def metric_row(frame: pd.DataFrame, group: dict[str, Any]) -> dict[str, Any]:
    error = frame["prediction"] - frame["target_return"]
    return {
        **group,
        "sample_count": int(len(frame)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "direction_accuracy": float(((frame["prediction"] >= 0) == (frame["target_return"] >= 0)).mean()),
        "prediction_std_mean": float(frame["prediction_std"].mean()),
    }


def grouped_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    grouper: Any = columns[0] if len(columns) == 1 else columns
    for key, group in frame.groupby(grouper, dropna=False, sort=True):
        values = (key,) if len(columns) == 1 else key
        rows.append(metric_row(group, dict(zip(columns, values))))
    return pd.DataFrame(rows)


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "fold_id", "trade_date", "stock_code", "target_return", "sample_valid", "text_available"]
    return predictions.groupby(keys, as_index=False, dropna=False).agg(
        prediction=("prediction", "mean"), prediction_std=("prediction", "std")
    ).fillna({"prediction_std": 0.0})


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_root = resolve(config["paths"]["adapter_root"])
    graphs = load_graphs(resolve(config["paths"]["graph_root"]))
    seeds = [int(seed) for seed in config["stability"]["seeds"]]

    receipts: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    stability: dict[str, Any] = {}
    for variant in config["controls"]:
        by_seed: dict[int, pd.DataFrame] = {}
        seed_mae: dict[int, float] = {}
        for seed in seeds:
            seed_config = copy.deepcopy(config)
            seed_config["seed"] = seed
            rows: list[dict[str, Any]] = []
            for fold_id in FOLD_IDS:
                data = load_fold(adapter_root, fold_id, variant["text_view"], bool(config["text"]["include_availability_and_log_count"]))
                receipt, fold_rows = run_fold(seed_config, variant, fold_id, data, graphs, output_root / "runs" / variant["id"] / f"seed_{seed}")
                receipt["seed"] = seed
                receipts.append(receipt)
                for row in fold_rows:
                    row["seed"] = seed
                rows.extend(fold_rows)
                print(f"{variant['id']} seed={seed} {fold_id} MAE={receipt['mae']:.6f}", flush=True)
            frame = pd.DataFrame(rows)
            by_seed[seed] = frame
            prediction_frames.append(frame)
            valid = frame["sample_valid"].astype(bool)
            seed_mae[seed] = float(np.abs(frame.loc[valid, "prediction"] - frame.loc[valid, "target_return"]).mean())

        pairwise = []
        for left_index in range(len(seeds)):
            for right_index in range(left_index + 1, len(seeds)):
                left, right = seeds[left_index], seeds[right_index]
                pairwise.append({
                    "seed_a": left,
                    "seed_b": right,
                    "edge_jaccard": 1.0,
                    "prediction_correlation": prediction_correlation(by_seed[left], by_seed[right]),
                })
        mae_values = np.asarray([seed_mae[seed] for seed in seeds], dtype=float)
        mae_cv = float(mae_values.std() / max(mae_values.mean(), 1e-12))
        thresholds = config["stability"]
        passed = (
            min(item["edge_jaccard"] for item in pairwise) >= float(thresholds["minimum_pairwise_edge_jaccard"])
            and min(item["prediction_correlation"] for item in pairwise) >= float(thresholds["minimum_pairwise_prediction_correlation"])
            and mae_cv <= float(thresholds["maximum_mae_coefficient_of_variation"])
        )
        stability[variant["id"]] = {
            "seed_mae": {str(key): value for key, value in seed_mae.items()},
            "pairwise": pairwise,
            "mae_coefficient_of_variation": mae_cv,
            "passed": passed,
        }

    fold_results_path = output_root / "fold_results.csv"
    pd.DataFrame(receipts).to_csv(fold_results_path, index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = output_root / "predictions.csv.gz"
    predictions.to_csv(predictions_path, index=False, compression={"method": "gzip", "mtime": 0})
    ensemble = ensemble_predictions(predictions)
    valid = ensemble.loc[ensemble["sample_valid"].astype(bool)].copy()
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    universe_columns = ["stock_code", "industry_group", "market_cap_bucket_cutoff"]
    valid = valid.merge(universe[universe_columns], on="stock_code", how="left", validate="many_to_one")
    valid["return_decile"] = valid.groupby("variant")["target_return"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    )

    diagnostic_paths = {}
    for name, columns in (
        ("per_stock", ["variant", "stock_code"]),
        ("industry", ["variant", "industry_group"]),
        ("market_cap", ["variant", "market_cap_bucket_cutoff"]),
        ("return_decile", ["variant", "return_decile"]),
    ):
        path = output_root / f"diagnostics_{name}.csv"
        grouped_metrics(valid, columns).to_csv(path, index=False)
        diagnostic_paths[name] = path

    pivot = valid.pivot(index=["fold_id", "trade_date", "stock_code"], columns="variant", values="prediction").dropna()
    control_ids = [item["id"] for item in config["controls"]]
    disagreement = pd.DataFrame({
        "fold_id": pivot.index.get_level_values("fold_id"),
        "trade_date": pivot.index.get_level_values("trade_date"),
        "stock_code": pivot.index.get_level_values("stock_code"),
        "absolute_prediction_difference": np.abs(pivot[control_ids[0]].to_numpy() - pivot[control_ids[1]].to_numpy()),
    })
    disagreement_path = output_root / "diagnostics_control_disagreement.csv"
    disagreement.to_csv(disagreement_path, index=False)

    original_config_path = resolve(config["paths"]["original_training_checks_config"])
    stable_controls = [variant for variant, result in stability.items() if result["passed"]]
    report = {
        "stage": "E-4 control closure",
        "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "controls": control_ids,
        "seeds": seeds,
        "thresholds": {key: config["stability"][key] for key in (
            "minimum_pairwise_edge_jaccard", "minimum_pairwise_prediction_correlation", "maximum_mae_coefficient_of_variation"
        )},
        "stability": stability,
        "stable_controls": stable_controls,
        "at_least_one_stable_control": bool(stable_controls),
        "allow_300_stock_graph_frequency_text_expansion": False,
        "future_or_sealed_data_read": False,
        "selection_exposure": "frozen fallback control evaluation; no candidate selection and no future data read",
        "config_sha256": sha256_file(config_path),
        "original_threshold_config_sha256": sha256_file(original_config_path),
        "artifacts": {
            "fold_results_sha256": sha256_file(fold_results_path),
            "predictions_sha256": sha256_file(predictions_path),
            **{f"diagnostics_{key}_sha256": sha256_file(path) for key, path in diagnostic_paths.items()},
            "diagnostics_control_disagreement_sha256": sha256_file(disagreement_path),
        },
    }
    report["batch_sha256"] = stable_json_sha256(report)
    (output_root / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(config_path, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
