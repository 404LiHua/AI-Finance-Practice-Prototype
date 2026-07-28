"""Run the frozen F-1.2 single-seed engineering receipt without ranking candidates."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from stage_e.e5.interface import E5FoldView, load_fold_view, validation_key_frame
from stage_e.e5.neural_graph import fixed_industry_adjacency
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import set_seed
from stage_f.custody import StageFDataCustodyGuard
from stage_f.robustness import (
    F1_CANDIDATE_IDS,
    F1CandidateStrategy,
    build_f1_candidate_model,
    candidate_training_step,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _validation_tensors(view: E5FoldView, sequence_length: int) -> tuple[torch.Tensor, ...]:
    indices = view.split_indices("validation")
    values = torch.from_numpy(view.numeric_values[indices, -sequence_length:].astype(np.float32))
    target_scaled = torch.from_numpy(view.target_scaled[indices].astype(np.float32))
    target_raw = torch.from_numpy(view.target_raw[indices].astype(np.float32))
    mask = torch.from_numpy(view.sample_mask[indices].astype(bool))
    return values, target_scaled, target_raw, mask


def train_candidate(
    candidate_id: str,
    view: E5FoldView,
    adjacency: np.ndarray,
    parameters: dict[str, Any],
    seed: int,
    checkpoint: Path,
    log_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate_id not in F1_CANDIDATE_IDS:
        raise ValueError(f"candidate outside frozen F-1.2 set: {candidate_id}")
    set_seed(seed)
    sequence_length = int(parameters["sequence_length"])
    train_indices = view.split_indices("train")
    train_x = view.numeric_values[train_indices, -sequence_length:].astype(np.float32)
    train_y_scaled = view.target_scaled[train_indices].astype(np.float32)
    train_y_raw = view.target_raw[train_indices].astype(np.float32)
    train_mask = view.sample_mask[train_indices].astype(bool)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y_scaled),
        torch.from_numpy(train_y_raw),
        torch.from_numpy(train_mask),
    )
    loader_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(parameters["batch_size"]),
        shuffle=True,
        generator=loader_generator,
    )
    model = build_f1_candidate_model(candidate_id, train_x.shape[-1], adjacency, parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    strategy = F1CandidateStrategy(candidate_id, seed)
    strategy.fit(train_y_raw, train_mask, "train")
    validation_x, validation_y_scaled, validation_y_raw, validation_mask = _validation_tensors(view, sequence_length)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, int(parameters["epochs"]) + 1):
        losses = []
        for batch_x, batch_y_scaled, batch_y_raw, batch_mask in loader:
            losses.append(candidate_training_step(
                model,
                optimizer,
                strategy,
                batch_x,
                batch_y_scaled,
                batch_y_raw,
                batch_mask,
                float(parameters["gradient_clip"]),
            ))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x)
            validation_loss_tensor = strategy.loss(
                validation_prediction, validation_y_scaled, validation_y_raw, validation_mask,
            )
        validation_loss = float(validation_loss_tensor)
        train_loss = float(np.mean(losses))
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise RuntimeError("non-finite F-1.2 training or validation loss")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = _state_copy(model)
            stale = 0
        else:
            stale += 1
        if stale >= int(parameters["patience"]):
            break
    if best_state is None:
        raise RuntimeError("F-1.2 did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    inference_started = time.perf_counter()
    with torch.no_grad():
        scaled = model(validation_x).cpu().numpy()
    inference_seconds = time.perf_counter() - inference_started
    prediction = scaled.reshape(-1) * float(view.target_std_train) + float(view.target_mean_train)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "candidate_id": candidate_id,
        "state_dict": model.state_dict(),
        "parameters": dict(parameters),
        "input_size": int(train_x.shape[-1]),
        "adjacency": adjacency.astype(np.float32),
        "target_mean": float(view.target_mean_train),
        "target_std": float(view.target_std_train),
        "tail_threshold_raw": strategy.tail_threshold_raw,
        "fold_id": view.fold_id,
        "seed": int(seed),
    }, checkpoint)
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return prediction.astype(float), {
        "training_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "first_train_loss": history[0]["train_loss"],
        "last_train_loss": history[-1]["train_loss"],
        "all_losses_finite": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tail_threshold_raw": strategy.tail_threshold_raw,
    }


def load_model(checkpoint: Path) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    candidate_id = str(payload["candidate_id"])
    model = build_f1_candidate_model(
        candidate_id,
        int(payload["input_size"]),
        np.asarray(payload["adjacency"], dtype=np.float32),
        dict(payload["parameters"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def load_predict_values(checkpoint: Path, values: np.ndarray) -> np.ndarray:
    model, payload = load_model(checkpoint)
    sequence_length = int(payload["parameters"]["sequence_length"])
    tensor = torch.from_numpy(values[:, -sequence_length:].astype(np.float32))
    with torch.no_grad():
        scaled = model(tensor).cpu().numpy()
    return scaled.astype(float) * float(payload["target_std"]) + float(payload["target_mean"])


def fit_observed_stress(view: E5FoldView, feature_order: list[str]) -> dict[str, float]:
    train = view.split_indices("train")
    feature_index = {name: index for index, name in enumerate(feature_order)}
    target = view.target_raw[train]
    valid = view.sample_mask[train]
    latest = view.numeric_values[train, -1]
    drawdown = view.numeric_values[train, -4:, :, feature_index["return_1w"]].sum(axis=1)
    return {
        "negative_return_tail_q10": float(np.quantile(target[valid], 0.10)),
        "positive_return_tail_q90": float(np.quantile(target[valid], 0.90)),
        "high_volatility_q90": float(np.quantile(latest[:, :, feature_index["return_vol_12"]][valid], 0.90)),
        "low_liquidity_q10": float(np.quantile(latest[:, :, feature_index["model_volume_hands"]][valid], 0.10)),
        "four_week_drawdown_q10": float(np.quantile(drawdown[valid], 0.10)),
    }


def stress_inference_receipt(
    checkpoint: Path,
    view: E5FoldView,
    seed: int,
    fold_index: int,
    feature_order: list[str],
    output_path: Path,
) -> dict[str, Any]:
    validation = view.split_indices("validation")
    values = view.numeric_values[validation].astype(np.float32)
    target = view.target_raw[validation].astype(np.float32)
    valid = view.sample_mask[validation].astype(bool)
    feature_index = {name: index for index, name in enumerate(feature_order)}
    thresholds = fit_observed_stress(view, feature_order)
    normal = load_predict_values(checkpoint, values)
    noise_rng = np.random.default_rng(int(seed) + 1101 + int(fold_index))
    noisy = values + noise_rng.normal(0.0, 0.05, size=values.shape).astype(np.float32)
    noise_prediction = load_predict_values(checkpoint, noisy)
    node_rng = np.random.default_rng(int(seed) + 2201 + int(fold_index))
    node_indices = np.sort(node_rng.choice(values.shape[2], size=max(1, round(values.shape[2] * 0.10)), replace=False))
    node_masked = values.copy()
    node_masked[:, :, node_indices, :] = 0.0
    node_prediction = load_predict_values(checkpoint, node_masked)
    latest_masked = values.copy()
    latest_masked[:, -1, :, :] = 0.0
    latest_prediction = load_predict_values(checkpoint, latest_masked)
    latest = values[:, -1]
    drawdown = values[:, -4:, :, feature_index["return_1w"]].sum(axis=1)
    scenario_masks = {
        "normal_unperturbed": valid,
        "negative_return_tail_q10": valid & (target <= thresholds["negative_return_tail_q10"]),
        "positive_return_tail_q90": valid & (target >= thresholds["positive_return_tail_q90"]),
        "high_volatility_q90": valid & (
            latest[:, :, feature_index["return_vol_12"]] >= thresholds["high_volatility_q90"]
        ),
        "low_liquidity_q10": valid & (
            latest[:, :, feature_index["model_volume_hands"]] <= thresholds["low_liquidity_q10"]
        ),
        "four_week_drawdown_q10": valid & (drawdown <= thresholds["four_week_drawdown_q10"]),
        "feature_noise_sigma_005": valid,
        "node_mask_10pct": valid,
        "latest_week_feature_mask": valid,
    }
    counts = {name: int(mask.sum()) for name, mask in scenario_masks.items()}
    if counts["normal_unperturbed"] <= 0 or any(
        counts[name] <= 0
        for name in ("feature_noise_sigma_005", "node_mask_10pct", "latest_week_feature_mask")
    ):
        raise RuntimeError(f"empty reference or synthetic F-1.2 stress entry: {counts}")
    arrays: dict[str, np.ndarray] = {
        "normal_prediction": normal,
        "feature_noise_prediction": noise_prediction,
        "node_mask_prediction": node_prediction,
        "latest_week_mask_prediction": latest_prediction,
        "target_raw": target,
        "sample_mask": valid,
        "masked_node_indices": node_indices.astype(np.int64),
    }
    for name, mask in scenario_masks.items():
        arrays[f"scenario_mask__{name}"] = mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    all_predictions = np.concatenate([
        normal.reshape(-1), noise_prediction.reshape(-1), node_prediction.reshape(-1), latest_prediction.reshape(-1),
    ])
    if not np.isfinite(all_predictions).all():
        raise RuntimeError("non-finite F-1.2 normal or stress prediction")
    return {
        "status": "PASS",
        "scenario_count": len(scenario_masks),
        "scenario_valid_sample_counts": counts,
        "train_only_observed_thresholds": thresholds,
        "synthetic_prediction_shapes": {
            "normal": list(normal.shape),
            "feature_noise_sigma_005": list(noise_prediction.shape),
            "node_mask_10pct": list(node_prediction.shape),
            "latest_week_feature_mask": list(latest_prediction.shape),
        },
        "masked_node_count": int(len(node_indices)),
        "empty_observed_scenarios_in_fold": sorted(
            name for name, count in counts.items()
            if count == 0 and name not in {
                "normal_unperturbed", "feature_noise_sigma_005", "node_mask_10pct", "latest_week_feature_mask"
            }
        ),
        "artifact_sha256": sha256_file(output_path),
    }


def _assert_frozen_keys(actual: pd.DataFrame, expected: pd.DataFrame, fold_id: str) -> str:
    columns = ["fold_id", "sample_row_id", "trade_date", "target_date", "stock_code", "sample_valid"]
    left = actual[columns].copy().astype({"stock_code": str})
    right = expected.loc[expected["fold_id"].astype(str) == fold_id, columns].copy().astype({"stock_code": str})
    left = left.sort_values(columns[:-1]).reset_index(drop=True)
    right = right.sort_values(columns[:-1]).reset_index(drop=True)
    if len(left) != 500 or len(right) != 500 or not left.equals(right):
        raise RuntimeError(f"{fold_id} validation sample keys differ from frozen E-5 keys")
    return stable_json_sha256(left.to_dict(orient="records"))


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["seed"]) != 20260725 or config["folds"] != ["E_RO_01", "E_RO_02", "E_RO_03"]:
        raise ValueError("F-1.2 seed and fold definitions are frozen")
    if tuple(config["candidate_ids"]) != F1_CANDIDATE_IDS:
        raise ValueError("F-1.2 candidate set or order changed")
    for source in config["source_registry"].values():
        if sha256_file(resolve(source["path"])) != source["sha256"]:
            raise RuntimeError(f"F-1.2 source hash mismatch: {source['path']}")
    guard = StageFDataCustodyGuard.from_config(resolve(config["paths"]["custody_config"]), REPO_ROOT)
    guarded_inputs = [
        resolve(config["paths"]["adapter_root"]),
        resolve(config["paths"]["universe_path"]),
        resolve(config["paths"]["frozen_validation_keys"]),
    ]
    guard.assert_paths_allowed(guarded_inputs, "f1_2_single_seed_engineering")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    expected_keys = pd.read_csv(resolve(config["paths"]["frozen_validation_keys"]), dtype={"stock_code": str})
    feature_order = list(config["feature_order"])
    parameters = dict(config["base_parameters"])
    seed = int(config["seed"])
    receipts = []
    failures = []
    predictions = []
    config_sha = sha256_file(config_path)
    for fold_index, fold_id in enumerate(config["folds"]):
        view = load_fold_view(resolve(config["paths"]["adapter_root"]), fold_id, "no_text")
        guard.assert_development_dates(view.trade_date, f"{fold_id} trade_date")
        guard.assert_development_dates(view.target_date.reshape(-1), f"{fold_id} target_date")
        if view.numeric_values.shape[-1] != len(feature_order) or len(view.stock_code) != 100:
            raise RuntimeError(f"{fold_id} frozen feature or stock dimension changed")
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
                    seed,
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
                    seed,
                    fold_index,
                    feature_order,
                    run_dir / "normal_and_stress_predictions.npz",
                )
                rows = key_frame.copy()
                rows.insert(0, "seed", seed)
                rows.insert(0, "candidate_id", candidate_id)
                rows["prediction"] = prediction
                predictions.append(rows)
                receipt = {
                    "candidate_id": candidate_id,
                    "fold_id": fold_id,
                    "seed": seed,
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
                receipts.append(receipt)
                (run_dir / "receipt.json").write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                print(f"{fold_id} {candidate_id} PASS load_diff={load_difference:.3g}", flush=True)
            except Exception as exc:
                failure = {
                    "candidate_id": candidate_id,
                    "fold_id": fold_id,
                    "seed": seed,
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "duration_seconds": time.perf_counter() - started,
                }
                failures.append(failure)
                (run_dir / "failure_receipt.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                print(f"{fold_id} {candidate_id} FAIL {type(exc).__name__}: {exc}", flush=True)
    receipt_path = output_root / "engineering_receipts.json"
    receipt_path.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failure_path = output_root / "failure_receipts.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prediction_path = output_root / "unified_predictions.csv.gz"
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not prediction_frame.empty:
        prediction_frame.to_csv(prediction_path, index=False, compression={"method": "gzip", "mtime": 0})
    pooled_stress_counts: dict[str, dict[str, int]] = {}
    for candidate_id in config["candidate_ids"]:
        candidate_receipts = [item for item in receipts if item["candidate_id"] == candidate_id]
        scenario_names = sorted({
            name
            for item in candidate_receipts
            for name in item["stress_inference"]["scenario_valid_sample_counts"]
        })
        pooled_stress_counts[candidate_id] = {
            name: sum(
                int(item["stress_inference"]["scenario_valid_sample_counts"].get(name, 0))
                for item in candidate_receipts
            )
            for name in scenario_names
        }
    pooled_stress_pass = len(pooled_stress_counts) == 3 and all(
        len(counts) == 9 and all(count > 0 for count in counts.values())
        for counts in pooled_stress_counts.values()
    )
    engineering_pass = len(receipts) == 9 and not failures and pooled_stress_pass
    metadata = {
        "stage": "F-1.2 single-seed engineering receipt",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if engineering_pass else "FAIL",
        "seed": seed,
        "folds": config["folds"],
        "candidate_ids": config["candidate_ids"],
        "expected_run_count": 9,
        "completed_run_count": len(receipts),
        "failure_count": len(failures),
        "all_losses_finite": bool(receipts) and all(item["all_losses_finite"] for item in receipts),
        "all_independent_loads_pass": bool(receipts) and all(
            item["independent_load_max_abs_difference"] <= config["independent_loading_max_abs_difference"]
            for item in receipts
        ),
        "all_stress_entries_pass": bool(receipts) and all(
            item["stress_inference"]["status"] == "PASS" and item["stress_inference"]["scenario_count"] == 9
            for item in receipts
        ) and pooled_stress_pass,
        "pooled_three_fold_stress_counts_by_candidate": pooled_stress_counts,
        "pooled_three_fold_stress_nonempty": pooled_stress_pass,
        "ranking_performed": False,
        "candidate_deletion_performed": False,
        "promotion_recommendation_formed": False,
        "additional_seed_executed": False,
        "gan_training_executed": False,
        "screening_accessed": False,
        "final_accessed": False,
        "config_sha256": config_sha,
        "artifacts": {
            "engineering_receipts_sha256": sha256_file(receipt_path),
            "failure_receipts_sha256": sha256_file(failure_path),
            "unified_predictions_sha256": sha256_file(prediction_path) if prediction_path.is_file() else None,
        },
    }
    metadata["batch_sha256"] = stable_json_sha256(metadata)
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not engineering_pass:
        raise RuntimeError("F-1.2 single-seed engineering did not complete all nine runs")
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
