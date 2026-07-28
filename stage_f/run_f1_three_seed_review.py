"""Complete F-1.3 by adding two authorized seeds without changing frozen training code."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage_e.e5.interface import load_fold_view, validation_key_frame
from stage_e.e5.neural_graph import fixed_industry_adjacency
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_f.custody import StageFDataCustodyGuard
from stage_f.robustness import F1_CANDIDATE_IDS
from stage_f.run_f1_single_seed import (
    _assert_frozen_keys,
    load_predict_values,
    stress_inference_receipt,
    train_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return 1.0 if np.allclose(left, right, rtol=0.0, atol=1e-12) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    return _correlation(left_rank, right_rank)


def stability_diagnostics(
    predictions: pd.DataFrame,
    candidate_ids: list[str],
    folds: list[str],
    seeds: list[int],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    key_columns = ["sample_row_id", "trade_date", "target_date", "stock_code"]
    models: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate = predictions.loc[predictions["candidate_id"].astype(str) == candidate_id].copy()
        fold_pairs = []
        all_prediction_matrices = []
        seed_mae = {}
        for seed in seeds:
            rows = candidate.loc[(candidate["seed"] == seed) & candidate["sample_valid"].astype(bool)]
            seed_mae[str(seed)] = float(np.mean(np.abs(rows["prediction"] - rows["target_return"])))
        for fold_id in folds:
            fold = candidate.loc[candidate["fold_id"].astype(str) == fold_id].copy()
            pivot = fold.pivot(index=key_columns, columns="seed", values="prediction").sort_index()
            if list(pivot.columns.astype(int)) != sorted(seeds) or len(pivot) != 500:
                raise RuntimeError(f"{candidate_id} {fold_id} three-seed prediction matrix is incomplete")
            matrix = pivot[sorted(seeds)].to_numpy(dtype=float)
            all_prediction_matrices.append(matrix)
            for left_seed, right_seed in itertools.combinations(sorted(seeds), 2):
                left = pivot[left_seed].to_numpy(dtype=float)
                right = pivot[right_seed].to_numpy(dtype=float)
                fold_pairs.append({
                    "fold_id": fold_id,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "pearson": _correlation(left, right),
                    "spearman": _spearman(left, right),
                })
        matrix = np.concatenate(all_prediction_matrices, axis=0)
        prediction_std = np.std(matrix, axis=1, ddof=0)
        mae_values = np.asarray(list(seed_mae.values()), dtype=float)
        mae_cv = float(np.std(mae_values, ddof=0) / max(float(np.mean(mae_values)), 1e-12))
        summary = {
            "candidate_id": candidate_id,
            "seed_mae": seed_mae,
            "seed_mae_cv": mae_cv,
            "minimum_pairwise_prediction_pearson": min(item["pearson"] for item in fold_pairs),
            "minimum_pairwise_prediction_spearman": min(item["spearman"] for item in fold_pairs),
            "prediction_seed_std_mean": float(np.mean(prediction_std)),
            "prediction_seed_std_p95": float(np.quantile(prediction_std, 0.95)),
            "fold_seed_pairs": fold_pairs,
        }
        gates = {
            "seed_mae_cv": summary["seed_mae_cv"] <= thresholds["seed_mae_cv_max"],
            "pairwise_pearson": summary["minimum_pairwise_prediction_pearson"]
            >= thresholds["minimum_all_pairwise_prediction_pearson"],
            "pairwise_spearman": summary["minimum_pairwise_prediction_spearman"]
            >= thresholds["minimum_all_pairwise_prediction_spearman"],
            "prediction_std_mean": summary["prediction_seed_std_mean"]
            <= thresholds["prediction_seed_std_mean_max"],
            "prediction_std_p95": summary["prediction_seed_std_p95"]
            <= thresholds["prediction_seed_std_p95_max"],
        }
        summary["stability_gates"] = gates
        summary["all_stability_gates_pass"] = all(gates.values())
        models.append(summary)
    return {
        "status": "COMPUTED_WITHOUT_RANKING",
        "thresholds": thresholds,
        "models_in_frozen_order": models,
        "ranking_performed": False,
        "promotion_recommendation_formed": False,
    }


def _pooled_stress_counts(receipts: list[dict[str, Any]], candidate_ids: list[str], seeds: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        output[candidate_id] = {}
        for seed in seeds:
            selected = [
                item for item in receipts
                if item["candidate_id"] == candidate_id and int(item["seed"]) == int(seed)
            ]
            names = sorted({
                name
                for item in selected
                for name in item["stress_inference"]["scenario_valid_sample_counts"]
            })
            counts = {
                name: sum(
                    int(item["stress_inference"]["scenario_valid_sample_counts"].get(name, 0))
                    for item in selected
                )
                for name in names
            }
            output[candidate_id][str(seed)] = counts
    return output


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["additional_seeds"] != [20260723, 20260724]:
        raise ValueError("F-1.3 may add only seeds 20260723 and 20260724")
    if config["all_seeds"] != [20260723, 20260724, 20260725]:
        raise ValueError("F-1.3 three-seed order changed")
    if tuple(config["candidate_ids"]) != F1_CANDIDATE_IDS:
        raise ValueError("F-1.3 candidate set or order changed")
    for source in config["source_registry"].values():
        if sha256_file(resolve(source["path"])) != source["sha256"]:
            raise RuntimeError(f"F-1.3 source hash mismatch: {source['path']}")
    upstream_config_path = resolve(config["upstream_f1_2"]["config_path"])
    upstream = json.loads(upstream_config_path.read_text(encoding="utf-8"))
    if sha256_file(upstream_config_path) != config["upstream_f1_2"]["config_sha256"]:
        raise RuntimeError("F-1.2 effective configuration hash changed")
    if upstream["seed"] != 20260725 or upstream["folds"] != config["folds"]:
        raise RuntimeError("F-1.2 upstream seed or folds changed")
    if upstream["candidate_ids"] != config["candidate_ids"] or upstream["base_parameters"] != config["base_parameters"]:
        raise RuntimeError("F-1.3 candidate set or training parameters differ from F-1.2")
    upstream_root = resolve(config["upstream_f1_2"]["output_root"])
    upstream_metadata = json.loads((upstream_root / "metadata.json").read_text(encoding="utf-8"))
    if upstream_metadata["status"] != "PASS" or upstream_metadata["completed_run_count"] != 9:
        raise RuntimeError("F-1.2 upstream engineering receipt is not accepted")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    guard = StageFDataCustodyGuard.from_config(resolve(upstream["paths"]["custody_config"]), REPO_ROOT)
    adapter_root = resolve(upstream["paths"]["adapter_root"])
    universe_path = resolve(upstream["paths"]["universe_path"])
    key_path = resolve(upstream["paths"]["frozen_validation_keys"])
    guard.assert_paths_allowed([adapter_root, universe_path, key_path], "f1_3_three_seed_review")
    universe = pd.read_csv(universe_path, dtype={"stock_code": str})
    expected_keys = pd.read_csv(key_path, dtype={"stock_code": str})
    feature_order = list(upstream["feature_order"])
    parameters = dict(upstream["base_parameters"])
    new_receipts = []
    failures = []
    new_predictions = []
    for seed in config["additional_seeds"]:
        for fold_index, fold_id in enumerate(config["folds"]):
            view = load_fold_view(adapter_root, fold_id, "no_text")
            guard.assert_development_dates(view.trade_date, f"{fold_id} trade_date")
            guard.assert_development_dates(view.target_date.reshape(-1), f"{fold_id} target_date")
            key_frame = validation_key_frame(view)
            key_sha = _assert_frozen_keys(key_frame, expected_keys, fold_id)
            adjacency, industries = fixed_industry_adjacency(view.stock_code, universe)
            stock_order_sha = stable_json_sha256(view.stock_code.astype(str).tolist())
            adjacency_sha = stable_json_sha256(adjacency.tolist())
            for candidate_id in config["candidate_ids"]:
                run_dir = output_root / "runs" / f"{fold_id}__{candidate_id}__seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                try:
                    checkpoint = run_dir / "model.pt"
                    prediction, detail = train_candidate(
                        candidate_id,
                        view,
                        adjacency,
                        parameters,
                        int(seed),
                        checkpoint,
                        run_dir / "training_log.json",
                    )
                    loaded = load_predict_values(
                        checkpoint,
                        view.numeric_values[view.split_indices("validation")],
                    ).reshape(-1)
                    load_difference = float(np.max(np.abs(prediction - loaded)))
                    if load_difference > float(config["independent_loading_max_abs_difference"]):
                        raise RuntimeError(f"independent load mismatch: {load_difference}")
                    stress = stress_inference_receipt(
                        checkpoint,
                        view,
                        int(seed),
                        fold_index,
                        feature_order,
                        run_dir / "normal_and_stress_predictions.npz",
                    )
                    rows = key_frame.copy()
                    rows.insert(0, "seed", int(seed))
                    rows.insert(0, "candidate_id", candidate_id)
                    rows["prediction"] = prediction
                    new_predictions.append(rows)
                    receipt = {
                        "candidate_id": candidate_id,
                        "fold_id": fold_id,
                        "seed": int(seed),
                        "status": "PASS",
                        "validation_rows": len(rows),
                        "validation_sample_key_sha256": key_sha,
                        "stock_order_sha256": stock_order_sha,
                        "adjacency_sha256": adjacency_sha,
                        "industry_count": len(set(industries)),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "independent_load_max_abs_difference": load_difference,
                        "stress_inference": stress,
                        "duration_seconds": time.perf_counter() - started,
                        **detail,
                    }
                    new_receipts.append(receipt)
                    (run_dir / "receipt.json").write_text(
                        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                    )
                    print(f"seed={seed} {fold_id} {candidate_id} PASS load_diff={load_difference:.3g}", flush=True)
                except Exception as exc:
                    failure = {
                        "candidate_id": candidate_id,
                        "fold_id": fold_id,
                        "seed": int(seed),
                        "status": "FAIL",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "duration_seconds": time.perf_counter() - started,
                    }
                    failures.append(failure)
                    (run_dir / "failure_receipt.json").write_text(
                        json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                    )
                    print(f"seed={seed} {fold_id} {candidate_id} FAIL {type(exc).__name__}: {exc}", flush=True)
    upstream_receipts = json.loads((upstream_root / "engineering_receipts.json").read_text(encoding="utf-8"))
    all_receipts = upstream_receipts + new_receipts
    upstream_predictions = pd.read_csv(
        upstream_root / "unified_predictions.csv.gz", dtype={"stock_code": str},
    )
    all_predictions = pd.concat(
        [upstream_predictions, pd.concat(new_predictions, ignore_index=True) if new_predictions else pd.DataFrame()],
        ignore_index=True,
    )
    expected_groups = len(config["candidate_ids"]) * len(config["folds"]) * len(config["all_seeds"])
    grouped = all_predictions.groupby(["candidate_id", "fold_id", "seed"]).size()
    contract_pass = len(all_predictions) == 13500 and len(grouped) == expected_groups and grouped.eq(500).all()
    stability = stability_diagnostics(
        all_predictions,
        config["candidate_ids"],
        config["folds"],
        config["all_seeds"],
        config["seed_stability_thresholds"],
    )
    pooled_stress = _pooled_stress_counts(all_receipts, config["candidate_ids"], config["all_seeds"])
    stress_pass = all(
        len(counts) == 9 and all(count > 0 for count in counts.values())
        for candidate in pooled_stress.values()
        for counts in candidate.values()
    )
    receipt_path = output_root / "additional_seed_receipts.json"
    receipt_path.write_text(json.dumps(new_receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_receipt_path = output_root / "all_three_seed_receipts.json"
    all_receipt_path.write_text(json.dumps(all_receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failure_path = output_root / "failure_receipts.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prediction_path = output_root / "three_seed_unified_predictions.csv.gz"
    all_predictions.to_csv(prediction_path, index=False, compression={"method": "gzip", "mtime": 0})
    stability_path = output_root / "three_seed_stability_diagnostics.json"
    stability_path.write_text(json.dumps(stability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stress_path = output_root / "three_seed_stress_counts.json"
    stress_path.write_text(json.dumps(pooled_stress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    engineering_pass = len(new_receipts) == 18 and not failures and len(all_receipts) == 27
    metadata = {
        "stage": "F-1.3 frozen three-seed review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if engineering_pass and contract_pass and stress_pass else "FAIL",
        "additional_seeds": config["additional_seeds"],
        "all_seeds": config["all_seeds"],
        "folds": config["folds"],
        "candidate_ids": config["candidate_ids"],
        "additional_run_count": len(new_receipts),
        "all_three_seed_run_count": len(all_receipts),
        "failure_count": len(failures),
        "prediction_rows": len(all_predictions),
        "prediction_contract_pass": bool(contract_pass),
        "all_losses_finite": bool(all_receipts) and all(item["all_losses_finite"] for item in all_receipts),
        "maximum_independent_load_difference": max(
            float(item["independent_load_max_abs_difference"]) for item in all_receipts
        ),
        "all_independent_loads_pass": all(
            float(item["independent_load_max_abs_difference"])
            <= float(config["independent_loading_max_abs_difference"])
            for item in all_receipts
        ),
        "three_seed_stress_nonempty": stress_pass,
        "stability_gate_results_in_frozen_candidate_order": [
            {
                "candidate_id": item["candidate_id"],
                "all_stability_gates_pass": item["all_stability_gates_pass"],
                "stability_gates": item["stability_gates"],
            }
            for item in stability["models_in_frozen_order"]
        ],
        "ranking_performed": False,
        "candidate_deletion_performed": False,
        "promotion_recommendation_formed": False,
        "gan_training_executed": False,
        "screening_accessed": False,
        "final_accessed": False,
        "config_sha256": sha256_file(config_path),
        "upstream_f1_2_config_sha256": config["upstream_f1_2"]["config_sha256"],
        "artifacts": {
            "additional_seed_receipts_sha256": sha256_file(receipt_path),
            "all_three_seed_receipts_sha256": sha256_file(all_receipt_path),
            "failure_receipts_sha256": sha256_file(failure_path),
            "three_seed_predictions_sha256": sha256_file(prediction_path),
            "stability_diagnostics_sha256": sha256_file(stability_path),
            "stress_counts_sha256": sha256_file(stress_path),
        },
    }
    metadata["batch_sha256"] = stable_json_sha256(metadata)
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if metadata["status"] != "PASS":
        raise RuntimeError("F-1.3 three-seed review failed engineering or contract checks")
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
