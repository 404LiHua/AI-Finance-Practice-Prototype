"""Run preregistered fixed-graph attribution and three-seed stability review."""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from stage_e.run_e4_ablations import load_fold, load_graphs, provided_adjacency, run_fold


FOLD_IDS = ("E_RO_01", "E_RO_02", "E_RO_03")


def summarize(receipts: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(receipts)
    return frame.groupby("variant", as_index=False).agg(
        mean_mae=("mae", "mean"),
        worst_fold_mae=("mae", "max"),
        fold_mae_std=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        mean_direction_accuracy=("direction_accuracy", "mean"),
        adjacency_valid=("adjacency_finite", "all"),
        row_stochastic=("adjacency_row_stochastic", "all"),
        parameter_count=("parameter_count", "max"),
        total_training_seconds=("training_seconds", "sum"),
    )


def select_variant(summary: pd.DataFrame) -> str:
    eligible = summary.loc[summary["adjacency_valid"].astype(bool) & summary["row_stochastic"].astype(bool)].copy()
    if eligible.empty:
        raise RuntimeError("no fixed-graph variant passed adjacency eligibility")
    ranked = eligible.sort_values(
        ["mean_mae", "worst_fold_mae", "fold_mae_std", "variant"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    return str(ranked.iloc[0]["variant"])


def prediction_correlation(left: pd.DataFrame, right: pd.DataFrame) -> float:
    keys = ["fold_id", "trade_date", "stock_code"]
    left_valid = left.loc[left["sample_valid"].astype(bool), keys + ["prediction"]].rename(columns={"prediction": "left"})
    right_valid = right.loc[right["sample_valid"].astype(bool), keys + ["prediction"]].rename(columns={"prediction": "right"})
    merged = left_valid.merge(right_valid, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left_valid) or len(merged) != len(right_valid):
        raise RuntimeError("three-seed predictions do not share the frozen validation key set")
    if merged["left"].std() == 0 or merged["right"].std() == 0:
        return 0.0
    return float(np.corrcoef(merged["left"], merged["right"])[0, 1])


def adjacency_edge_mask(variant: dict[str, str], fold_id: str, config: dict[str, Any], graphs: dict[str, Any]) -> np.ndarray:
    data = load_fold(resolve(config["paths"]["adapter_root"]), fold_id, variant["text_view"], bool(config["text"]["include_availability_and_log_count"]))
    validation_indices = np.flatnonzero(data["split"] == "validation")
    adjacency = provided_adjacency(variant["graph"], validation_indices, data["dates"], graphs)
    if adjacency is None:
        raise RuntimeError("stability candidate must use a provided fixed graph")
    values = adjacency.numpy()
    diagonal = np.eye(values.shape[-1], dtype=bool)[None, :, :]
    return (values > 0) & ~diagonal


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_root = resolve(config["paths"]["adapter_root"])
    graph_root = resolve(config["paths"]["graph_root"])
    graphs = load_graphs(graph_root)

    development_receipts: list[dict[str, Any]] = []
    development_predictions: list[dict[str, Any]] = []
    for variant in config["variants"]:
        for fold_id in FOLD_IDS:
            data = load_fold(adapter_root, fold_id, variant["text_view"], bool(config["text"]["include_availability_and_log_count"]))
            receipt, rows = run_fold(config, variant, fold_id, data, graphs, output_root / "development")
            development_receipts.append(receipt)
            development_predictions.extend(rows)
            print(f"development {variant['id']} {fold_id} MAE={receipt['mae']:.6f}", flush=True)

    development_fold_path = output_root / "development_fold_results.csv"
    pd.DataFrame(development_receipts).to_csv(development_fold_path, index=False)
    development_summary = summarize(development_receipts)
    development_summary_path = output_root / "development_summary.csv"
    development_summary.sort_values(["mean_mae", "worst_fold_mae", "fold_mae_std", "variant"], kind="stable").to_csv(development_summary_path, index=False)
    development_predictions_path = output_root / "development_predictions.csv.gz"
    pd.DataFrame(development_predictions).to_csv(development_predictions_path, index=False, compression={"method": "gzip", "mtime": 0})
    selected_id = select_variant(development_summary)
    selected = next(copy.deepcopy(item) for item in config["variants"] if item["id"] == selected_id)

    stability_receipts: list[dict[str, Any]] = []
    predictions_by_seed: dict[int, pd.DataFrame] = {}
    seed_maes: dict[int, float] = {}
    for seed in config["stability"]["seeds"]:
        seed_config = copy.deepcopy(config)
        seed_config["seed"] = int(seed)
        seed_rows: list[dict[str, Any]] = []
        for fold_id in FOLD_IDS:
            data = load_fold(adapter_root, fold_id, selected["text_view"], bool(config["text"]["include_availability_and_log_count"]))
            receipt, rows = run_fold(seed_config, selected, fold_id, data, graphs, output_root / "stability" / f"seed_{seed}")
            receipt["seed"] = int(seed)
            stability_receipts.append(receipt)
            for row in rows:
                row["seed"] = int(seed)
            seed_rows.extend(rows)
            print(f"stability seed={seed} {fold_id} MAE={receipt['mae']:.6f}", flush=True)
        frame = pd.DataFrame(seed_rows)
        predictions_by_seed[int(seed)] = frame
        valid = frame["sample_valid"].astype(bool)
        seed_maes[int(seed)] = float(np.mean(np.abs(frame.loc[valid, "prediction"] - frame.loc[valid, "target_return"])))

    stability_fold_path = output_root / "stability_fold_results.csv"
    pd.DataFrame(stability_receipts).to_csv(stability_fold_path, index=False)
    stability_predictions_path = output_root / "stability_predictions.csv.gz"
    pd.concat(predictions_by_seed.values(), ignore_index=True).to_csv(stability_predictions_path, index=False, compression={"method": "gzip", "mtime": 0})

    seeds = [int(seed) for seed in config["stability"]["seeds"]]
    edge_masks = {fold_id: adjacency_edge_mask(selected, fold_id, config, graphs) for fold_id in FOLD_IDS}
    pairwise: list[dict[str, Any]] = []
    for left_index in range(len(seeds)):
        for right_index in range(left_index + 1, len(seeds)):
            left, right = seeds[left_index], seeds[right_index]
            intersections = sum(np.logical_and(mask, mask).sum() for mask in edge_masks.values())
            unions = sum(np.logical_or(mask, mask).sum() for mask in edge_masks.values())
            pairwise.append({
                "seed_a": left,
                "seed_b": right,
                "edge_jaccard": float(intersections / unions) if unions else 1.0,
                "prediction_correlation": prediction_correlation(predictions_by_seed[left], predictions_by_seed[right]),
            })
    mae_values = np.asarray([seed_maes[seed] for seed in seeds], dtype=float)
    mae_cv = float(mae_values.std() / max(mae_values.mean(), 1e-12))
    thresholds = config["stability"]
    stability_pass = (
        all(row["adjacency_finite"] and row["adjacency_row_stochastic"] for row in stability_receipts)
        and min(row["edge_jaccard"] for row in pairwise) >= float(thresholds["minimum_pairwise_edge_jaccard"])
        and min(row["prediction_correlation"] for row in pairwise) >= float(thresholds["minimum_pairwise_prediction_correlation"])
        and mae_cv <= float(thresholds["maximum_mae_coefficient_of_variation"])
    )

    original_config_path = resolve(config["paths"]["original_training_checks_config"])
    report = {
        "stage": "E-4.4 fixed-graph stabilization",
        "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_seed": config["seed"],
        "development_variant_count": len(config["variants"]),
        "development_fold_run_count": len(development_receipts),
        "selection_rule": config["selection_rule"],
        "selected_variant": selected_id,
        "stability_seeds": seeds,
        "seed_mae": {str(key): value for key, value in seed_maes.items()},
        "pairwise": pairwise,
        "mae_coefficient_of_variation": mae_cv,
        "thresholds": {key: thresholds[key] for key in (
            "minimum_pairwise_edge_jaccard", "minimum_pairwise_prediction_correlation", "maximum_mae_coefficient_of_variation"
        )},
        "stability_pass": stability_pass,
        "allow_300_stock_expansion": stability_pass,
        "fallback_controls": [] if stability_pass else config["fallback_controls"],
        "selection_exposure": "TRAIN/VALIDATION fixed-graph stabilization; no future data read and no free candidate added",
        "future_or_sealed_data_read": False,
        "config_sha256": sha256_file(config_path),
        "original_threshold_config_sha256": sha256_file(original_config_path),
        "fixed_graphs_sha256": sha256_file(graphs["fixed_path"]),
        "rolling_graphs_sha256": sha256_file(graphs["rolling_path"]),
        "artifacts": {
            "development_fold_results_sha256": sha256_file(development_fold_path),
            "development_summary_sha256": sha256_file(development_summary_path),
            "development_predictions_sha256": sha256_file(development_predictions_path),
            "stability_fold_results_sha256": sha256_file(stability_fold_path),
            "stability_predictions_sha256": sha256_file(stability_predictions_path)
        }
    }
    report["batch_sha256"] = stable_json_sha256(report)
    report_path = output_root / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
