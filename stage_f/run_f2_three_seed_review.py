"""Add only seeds 20260723/20260724 under the unchanged accepted F-2.2 V3 training code."""

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
import torch

from stage_e.e5.interface import load_fold_view, validation_key_frame
from stage_e.e5.neural_graph import fixed_industry_adjacency
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_f.custody import StageFDataCustodyGuard
from stage_f.run_f2_single_seed import (
    assert_frozen_keys,
    load_frozen_forecaster,
    load_predict_values,
    stress_inference,
    train_augmented_forecaster,
    train_gan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return 1.0 if np.allclose(left, right, rtol=0.0, atol=1e-12) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    return correlation(left_rank, right_rank)


def stability_diagnostics(
    predictions: pd.DataFrame, folds: list[str], seeds: list[int], thresholds: dict[str, float],
) -> dict[str, Any]:
    key_columns = ["sample_row_id", "trade_date", "target_date", "stock_code"]
    seed_mae: dict[str, float] = {}
    for seed in seeds:
        rows = predictions.loc[(predictions["seed"] == seed) & predictions["sample_valid"].astype(bool)]
        seed_mae[str(seed)] = float(np.mean(np.abs(rows["prediction"] - rows["target_return"])))
    pair_rows: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    for fold_id in folds:
        fold = predictions.loc[predictions["fold_id"].astype(str) == fold_id]
        pivot = fold.pivot(index=key_columns, columns="seed", values="prediction").sort_index()
        if list(pivot.columns.astype(int)) != sorted(seeds) or len(pivot) != 500:
            raise RuntimeError(f"F-2.3 {fold_id} three-seed prediction matrix incomplete")
        matrix = pivot[sorted(seeds)].to_numpy(dtype=float)
        matrices.append(matrix)
        for left_seed, right_seed in itertools.combinations(sorted(seeds), 2):
            left = pivot[left_seed].to_numpy(dtype=float)
            right = pivot[right_seed].to_numpy(dtype=float)
            pair_rows.append({
                "fold_id": fold_id, "left_seed": left_seed, "right_seed": right_seed,
                "pearson": correlation(left, right), "spearman": spearman(left, right),
            })
    matrix = np.concatenate(matrices, axis=0)
    prediction_std = np.std(matrix, axis=1, ddof=0)
    mae_values = np.asarray(list(seed_mae.values()), dtype=float)
    summary = {
        "candidate_id": "stock_node_gwnet_bounded_cwgan_gp_l8",
        "seed_mae": seed_mae,
        "seed_mae_cv": float(np.std(mae_values, ddof=0) / max(float(np.mean(mae_values)), 1e-12)),
        "minimum_pairwise_prediction_pearson": min(row["pearson"] for row in pair_rows),
        "minimum_pairwise_prediction_spearman": min(row["spearman"] for row in pair_rows),
        "prediction_seed_std_mean": float(np.mean(prediction_std)),
        "prediction_seed_std_p95": float(np.quantile(prediction_std, 0.95)),
        "fold_seed_pairs": pair_rows,
    }
    gates = {
        "seed_mae_cv": summary["seed_mae_cv"] <= thresholds["seed_mae_cv_max"],
        "pairwise_pearson": summary["minimum_pairwise_prediction_pearson"]
        >= thresholds["minimum_all_pairwise_prediction_pearson"],
        "pairwise_spearman": summary["minimum_pairwise_prediction_spearman"]
        >= thresholds["minimum_all_pairwise_prediction_spearman"],
        "prediction_std_mean": summary["prediction_seed_std_mean"] <= thresholds["prediction_seed_std_mean_max"],
        "prediction_std_p95": summary["prediction_seed_std_p95"] <= thresholds["prediction_seed_std_p95_max"],
    }
    summary["stability_gates"] = gates
    summary["all_stability_gates_pass"] = all(gates.values())
    return summary


def pooled_stress_counts(receipts: list[dict[str, Any]], seeds: list[int]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for seed in seeds:
        selected = [receipt for receipt in receipts if int(receipt["seed"]) == seed]
        names = sorted({
            name for receipt in selected for name in receipt["stress_inference"]["scenario_valid_sample_counts"]
        })
        result[str(seed)] = {
            name: sum(int(receipt["stress_inference"]["scenario_valid_sample_counts"].get(name, 0)) for receipt in selected)
            for name in names
        }
    return result


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["additional_seeds"] != [20260723, 20260724]:
        raise ValueError("F-2.3 may add only seeds 20260723 and 20260724")
    if config["all_seeds"] != [20260723, 20260724, 20260725]:
        raise ValueError("F-2.3 all-seed order changed")
    for source in config["source_registry"].values():
        if sha256_file(resolve(source["path"])) != source["sha256"]:
            raise RuntimeError(f"F-2.3 source hash mismatch: {source['path']}")
    base_config_path = resolve(config["upstream_f2_2"]["base_config_path"])
    if sha256_file(base_config_path) != config["upstream_f2_2"]["base_config_sha256"]:
        raise RuntimeError("F-2.2 frozen base config hash changed")
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if sha256_file(resolve(config["upstream_f2_2"]["effective_config_path"])) != config["upstream_f2_2"]["effective_config_sha256"]:
        raise RuntimeError("F-2.2 effective V3 config hash changed")
    upstream_root = resolve(config["upstream_f2_2"]["output_root"])
    upstream_metadata = json.loads((upstream_root / "metadata.json").read_text(encoding="utf-8"))
    if upstream_metadata["status"] != "PASS" or upstream_metadata["completed_run_count"] != 3:
        raise RuntimeError("F-2.2 seed 20260725 upstream is not accepted")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    guard = StageFDataCustodyGuard.from_config(resolve(base_config["paths"]["custody_config"]), REPO_ROOT)
    adapter_root = resolve(base_config["paths"]["adapter_root"])
    universe_path = resolve(base_config["paths"]["universe_path"])
    key_path = resolve(base_config["paths"]["frozen_validation_keys"])
    guard.assert_paths_allowed([adapter_root, universe_path, key_path], "f2_3_three_seed_review")
    universe = pd.read_csv(universe_path, dtype={"stock_code": str})
    expected_keys = pd.read_csv(key_path, dtype={"stock_code": str})
    new_receipts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    new_predictions: list[pd.DataFrame] = []
    for seed in config["additional_seeds"]:
        for fold_index, fold_id in enumerate(config["folds"]):
            run_dir = output_root / "runs" / f"{fold_id}__{config['candidate_id']}__seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                view = load_fold_view(adapter_root, fold_id, "no_text")
                guard.assert_development_dates(view.trade_date, f"{fold_id} trade_date")
                guard.assert_development_dates(view.target_date.reshape(-1), f"{fold_id} target_date")
                key_frame = validation_key_frame(view)
                key_sha = assert_frozen_keys(key_frame, expected_keys, fold_id)
                adjacency, industries = fixed_industry_adjacency(view.stock_code, universe)
                frozen_item = config["frozen_forecaster_checkpoints"][str(seed)][fold_id]
                frozen_path = resolve(frozen_item["path"])
                if sha256_file(frozen_path) != frozen_item["sha256"]:
                    raise RuntimeError(f"frozen forecaster hash mismatch: seed={seed} {fold_id}")
                frozen_forecaster, _ = load_frozen_forecaster(frozen_path)
                run_config = json.loads(json.dumps(base_config))
                run_config["seed"] = int(seed)
                generator, critic, gan_detail = train_gan(
                    view, frozen_forecaster, run_config, int(seed), run_dir / "gan_training_log.json",
                )
                gan_checkpoint = run_dir / "gan_final_epoch.pt"
                torch.save({
                    "candidate_id": config["candidate_id"], "fold_id": fold_id, "seed": int(seed),
                    "generator_state_dict": generator.state_dict(), "critic_state_dict": critic.state_dict(),
                    "generator_constructor": {"feature_count": 6, "noise_dim": 8, "hidden_channels": 32, "max_delta": 0.05},
                    "critic_constructor": {"feature_count": 6, "hidden_channels": 32},
                }, gan_checkpoint)
                forecaster_checkpoint = run_dir / "forecaster.pt"
                prediction, forecaster_detail = train_augmented_forecaster(
                    view, adjacency, generator, run_config, int(seed), forecaster_checkpoint,
                    run_dir / "forecaster_training_log.json",
                )
                loaded = load_predict_values(
                    forecaster_checkpoint, view.numeric_values[view.split_indices("validation")],
                ).reshape(-1)
                load_difference = float(np.max(np.abs(prediction - loaded)))
                if load_difference > float(config["independent_loading_max_abs_difference"]):
                    raise RuntimeError(f"independent load mismatch: {load_difference}")
                stress = stress_inference(
                    forecaster_checkpoint, view, int(seed), fold_index, list(base_config["feature_order"]),
                    run_dir / "normal_and_stress_predictions.npz",
                )
                duration = time.perf_counter() - started
                if duration > float(config["maximum_fold_duration_seconds"]):
                    raise RuntimeError(f"F-2.3 fold duration cost hard failure: {duration}")
                rows = key_frame.copy()
                rows.insert(0, "seed", int(seed))
                rows.insert(0, "candidate_id", config["candidate_id"])
                rows["prediction"] = prediction
                new_predictions.append(rows)
                receipt = {
                    "candidate_id": config["candidate_id"], "fold_id": fold_id, "seed": int(seed), "status": "PASS",
                    "validation_rows": len(rows), "validation_sample_key_sha256": key_sha,
                    "stock_order_sha256": stable_json_sha256(view.stock_code.astype(str).tolist()),
                    "adjacency_sha256": stable_json_sha256(adjacency.tolist()), "industry_count": len(set(industries)),
                    "frozen_forecaster_sha256": sha256_file(frozen_path),
                    "gan_checkpoint_sha256": sha256_file(gan_checkpoint),
                    "forecaster_checkpoint_sha256": sha256_file(forecaster_checkpoint),
                    "independent_load_max_abs_difference": load_difference,
                    "stress_inference": stress, "duration_seconds": duration, **gan_detail, **forecaster_detail,
                }
                new_receipts.append(receipt)
                (run_dir / "receipt.json").write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                print(f"seed={seed} {fold_id} PASS duration={duration:.3f}s load_diff={load_difference:.3g}", flush=True)
            except Exception as exc:
                failure = {
                    "candidate_id": config["candidate_id"], "fold_id": fold_id, "seed": int(seed), "status": "FAIL",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "collapse_conditions": getattr(exc, "conditions", []),
                    "duration_seconds": time.perf_counter() - started,
                }
                failures.append(failure)
                (run_dir / "failure_receipt.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                print(f"seed={seed} {fold_id} FAIL {type(exc).__name__}: {exc}", flush=True)
    upstream_receipts = json.loads((upstream_root / "engineering_receipts.json").read_text(encoding="utf-8"))
    all_receipts = new_receipts + upstream_receipts
    upstream_predictions = pd.read_csv(upstream_root / "unified_predictions.csv.gz", dtype={"stock_code": str})
    appended = pd.concat(new_predictions, ignore_index=True) if new_predictions else pd.DataFrame()
    all_predictions = pd.concat([appended, upstream_predictions], ignore_index=True)
    grouped = all_predictions.groupby(["candidate_id", "fold_id", "seed"]).size()
    prediction_contract_pass = len(all_predictions) == 4500 and len(grouped) == 9 and grouped.eq(500).all()
    stability = stability_diagnostics(
        all_predictions, config["folds"], config["all_seeds"], config["seed_stability_thresholds"],
    )
    stress_counts = pooled_stress_counts(all_receipts, config["all_seeds"])
    stress_pass = all(len(counts) == 9 and all(value > 0 for value in counts.values()) for counts in stress_counts.values())
    total_nine_run_seconds = sum(float(receipt["duration_seconds"]) for receipt in all_receipts)
    cost_pass = all(
        float(receipt["duration_seconds"]) <= float(config["maximum_fold_duration_seconds"])
        for receipt in all_receipts
    ) and total_nine_run_seconds <= float(config["maximum_total_nine_run_seconds"])
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
    stress_path.write_text(json.dumps(stress_counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    engineering_pass = (
        len(new_receipts) == 6 and len(all_receipts) == 9 and not failures and prediction_contract_pass
        and stress_pass and cost_pass
        and all(receipt["all_gan_losses_finite"] and receipt["all_forecaster_losses_finite"] for receipt in all_receipts)
        and all(receipt["collapse_conditions_pass"] for receipt in all_receipts)
        and all(float(receipt["independent_load_max_abs_difference"]) <= config["independent_loading_max_abs_difference"]
                for receipt in all_receipts)
    )
    metadata = {
        "stage": "F-2.3 frozen three-seed GAN review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if engineering_pass else "FAIL",
        "candidate_id": config["candidate_id"], "additional_seeds": config["additional_seeds"],
        "all_seeds": config["all_seeds"], "folds": config["folds"],
        "additional_run_count": len(new_receipts), "all_three_seed_run_count": len(all_receipts),
        "failure_count": len(failures), "prediction_rows": len(all_predictions),
        "prediction_contract_pass": bool(prediction_contract_pass),
        "all_losses_finite": bool(all_receipts) and all(
            receipt["all_gan_losses_finite"] and receipt["all_forecaster_losses_finite"] for receipt in all_receipts
        ),
        "all_collapse_conditions_pass": bool(all_receipts) and all(
            receipt["collapse_conditions_pass"] for receipt in all_receipts
        ),
        "maximum_independent_load_difference": max(
            float(receipt["independent_load_max_abs_difference"]) for receipt in all_receipts
        ),
        "all_independent_loads_pass": bool(all_receipts) and all(
            float(receipt["independent_load_max_abs_difference"]) <= config["independent_loading_max_abs_difference"]
            for receipt in all_receipts
        ),
        "three_seed_stress_nonempty": stress_pass,
        "total_nine_run_seconds": total_nine_run_seconds, "cost_limit_pass": cost_pass,
        "stability_result": stability,
        "ranking_performed": False, "candidate_deletion_performed": False,
        "promotion_recommendation_formed": False, "screening_accessed": False, "final_accessed": False,
        "config_sha256": sha256_file(config_path),
        "upstream_f2_2_config_sha256": config["upstream_f2_2"]["effective_config_sha256"],
        "artifacts": {
            "additional_seed_receipts_sha256": sha256_file(receipt_path),
            "all_three_seed_receipts_sha256": sha256_file(all_receipt_path),
            "failure_receipts_sha256": sha256_file(failure_path),
            "three_seed_predictions_sha256": sha256_file(prediction_path),
            "stability_diagnostics_sha256": sha256_file(stability_path),
            "stress_counts_sha256": sha256_file(stress_path),
        },
        "next_action": config["next_action_if_complete"],
    }
    metadata["batch_sha256"] = stable_json_sha256(metadata)
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not engineering_pass:
        raise RuntimeError("F-2.3 three-seed GAN review failed engineering checks")
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
