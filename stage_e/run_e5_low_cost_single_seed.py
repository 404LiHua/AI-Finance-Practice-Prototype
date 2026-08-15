"""Run the preregistered E-5.2 low-cost baselines for one engineering seed."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.e5.evaluation import evaluate_predictions, validate_prediction_contract
from stage_e.e5.interface import load_fold_view, validation_key_frame
from stage_e.e5.low_cost import (
    load_predict_industry_var, load_predict_sklearn, load_predict_torch,
    train_industry_var, train_sklearn_model, train_torch_model,
)
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


def run(
    config_path: Path,
    overwrite: bool = False,
    seed_override: int | None = None,
    output_root_override: Path | None = None,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "PREREGISTERED_LOCKED_BEFORE_SINGLE_SEED_RUN":
        raise ValueError("E-5.2 model set must be locked before running")
    interface_path = resolve(config["interface_config"])
    if sha256_file(interface_path) != config["interface_config_sha256"]:
        raise RuntimeError("frozen E-5 interface config hash mismatch")
    interface = json.loads(interface_path.read_text(encoding="utf-8"))
    if config["restrictions"]["candidate_selection_allowed"] or config["restrictions"]["model_deletion_allowed"]:
        raise ValueError("single-seed engineering cannot select or delete models")
    output_root = output_root_override or resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_root = resolve(config["paths"]["adapter_root"])
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    frets_source = resolve(config["paths"]["frets_source"])
    if not frets_source.is_file():
        raise FileNotFoundError(frets_source)
    config_sha = sha256_file(config_path)
    seed = int(config["engineering_seed"] if seed_override is None else seed_override)
    allowed_seeds = {int(value) for value in config["future_three_seeds"]}
    if seed not in allowed_seeds:
        raise ValueError(f"seed override is outside the preregistered seed set: {seed}")
    all_predictions = []
    receipts = []
    failures = []
    expected_keys = []

    for fold_id in config["folds"]:
        view = load_fold_view(adapter_root, fold_id, "no_text")
        key_frame = validation_key_frame(view)
        expected_keys.append(key_frame)
        stock_meta = pd.DataFrame({"stock_code": view.stock_code.astype(str)}).merge(
            universe[["stock_code", "industry_group"]], on="stock_code", how="left", validate="one_to_one"
        )
        industries, industry_index = np.unique(stock_meta["industry_group"].fillna("UNKNOWN").astype(str), return_inverse=True)
        for model_spec in config["models"]:
            model_id = str(model_spec["id"])
            family = str(model_spec["family"])
            parameters = dict(model_spec["parameters"])
            run_dir = output_root / "runs" / f"{fold_id}__{model_id}__seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            resolved_path = run_dir / "resolved_config.json"
            resolved_path.write_text(json.dumps({
                "experiment_id": config["experiment_id"], "model_id": model_id, "family": family,
                "feature_view": model_spec["feature_view"], "parameters": parameters,
                "fold_id": fold_id, "seed": seed, "interface_config_sha256": config["interface_config_sha256"],
                "selection_exposure": config["selection_exposure"], "future_or_sealed_data_read": False,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            started = time.perf_counter()
            try:
                if family == "deterministic":
                    prediction = np.full(len(key_frame), float(parameters["forecast_return"]), dtype=float)
                    checkpoint = run_dir / "model.json"
                    checkpoint.write_text(json.dumps({"family": family, **parameters}, indent=2) + "\n", encoding="utf-8")
                    loaded = np.full(len(key_frame), json.loads(checkpoint.read_text(encoding="utf-8"))["forecast_return"], dtype=float)
                    detail = {"training_seconds": 0.0, "parameter_count": 0}
                elif family in {"frets", "minimalist_transformer"}:
                    checkpoint = run_dir / "model.pt"
                    prediction, detail = train_torch_model(
                        family, view, parameters, seed, frets_source if family == "frets" else None,
                        checkpoint, run_dir / "training_log.json",
                    )
                    loaded = load_predict_torch(checkpoint, view, frets_source if family == "frets" else None)
                elif family in {"random_forest", "svm_rbf"}:
                    checkpoint = run_dir / "model.joblib"
                    prediction, detail = train_sklearn_model(family, view, parameters, seed, checkpoint)
                    loaded = load_predict_sklearn(checkpoint, view)
                    (run_dir / "training_log.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
                elif family == "industry_var1_ridge":
                    checkpoint = run_dir / "model.npz"
                    prediction, detail = train_industry_var(
                        view, industry_index.astype(np.int64), industries.astype(str),
                        float(parameters["ridge_alpha"]), checkpoint,
                    )
                    loaded = load_predict_industry_var(checkpoint, view)
                    (run_dir / "training_log.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
                else:
                    raise ValueError(f"unsupported locked family: {family}")
                maximum_load_difference = float(np.max(np.abs(prediction - loaded)))
                if maximum_load_difference > float(config["independent_loading"]["maximum_prediction_absolute_difference"]):
                    raise RuntimeError(f"independent load prediction mismatch: {maximum_load_difference}")
                if not np.isfinite(prediction).all():
                    raise RuntimeError("non-finite baseline prediction")
                checkpoint_sha = sha256_file(checkpoint)
                rows = key_frame.copy()
                rows.insert(0, "seed", seed)
                rows.insert(0, "model_id", model_id)
                rows["prediction"] = prediction
                rows["checkpoint_sha256"] = checkpoint_sha
                rows["config_sha256"] = config_sha
                all_predictions.append(rows)
                receipt = {
                    "model_id": model_id, "family": family, "fold_id": fold_id, "seed": seed,
                    "status": "PASS", "prediction_rows": len(rows),
                    "independent_load_max_abs_difference": maximum_load_difference,
                    "checkpoint_sha256": checkpoint_sha, "resolved_config_sha256": sha256_file(resolved_path),
                    "duration_seconds": time.perf_counter() - started, **detail,
                }
                receipts.append(receipt)
                (run_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"{fold_id} {model_id} PASS rows={len(rows)} load_diff={maximum_load_difference:.3g}", flush=True)
            except Exception as exc:
                failure = {
                    "model_id": model_id, "family": family, "fold_id": fold_id, "seed": seed,
                    "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
                    "duration_seconds": time.perf_counter() - started,
                }
                failures.append(failure)
                (run_dir / "failure_receipt.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"{fold_id} {model_id} FAIL {type(exc).__name__}: {exc}", flush=True)

    expected = pd.concat(expected_keys, ignore_index=True)
    expected_path = output_root / "frozen_validation_keys.csv.gz"
    expected.to_csv(expected_path, index=False, compression={"method": "gzip", "mtime": 0})
    receipts_path = output_root / "engineering_receipts.csv"
    pd.DataFrame(receipts).to_csv(receipts_path, index=False)
    failures_path = output_root / "failure_receipts.json"
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prediction_path = output_root / "unified_predictions.csv.gz"
    if all_predictions:
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions.to_csv(prediction_path, index=False, compression={"method": "gzip", "mtime": 0})
        # The persisted prediction contract is the canonical evaluator input.  Re-reading it
        # prevents in-memory float precision from making independent metric recomputation differ.
        predictions = pd.read_csv(prediction_path, dtype={"stock_code": str})
    else:
        predictions = pd.DataFrame()

    contract_receipt: dict[str, Any] | None = None
    evaluation_hashes = {}
    if not predictions.empty:
        completed_models = sorted(predictions["model_id"].astype(str).unique())
        contract_receipt = validate_prediction_contract(
            predictions, expected, config["folds"], [seed],
            require_all_frozen_keys=True, allow_extra_rows=False,
        )
        evaluations = evaluate_predictions(
            predictions, universe, float(interface["evaluation"]["mape_denominator_floor"]),
            float(interface["evaluation"]["direction_positive_threshold"]), int(interface["evaluation"]["return_group_count"]),
        )
        for name, frame in evaluations.items():
            path = output_root / f"{name}.csv"
            frame.to_csv(path, index=False)
            evaluation_hashes[f"{name}_sha256"] = sha256_file(path)
    else:
        completed_models = []
    expected_models = [str(item["id"]) for item in config["models"]]
    engineering_pass = not failures and completed_models == sorted(expected_models) and len(receipts) == len(expected_models) * len(config["folds"])
    metadata = {
        "stage": "E-5.2 single-seed engineering", "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "engineering_seed": seed,
        "expected_models": expected_models, "completed_models": completed_models,
        "run_count": len(receipts), "failure_count": len(failures), "engineering_pass": engineering_pass,
        "candidate_selection_performed": False, "model_deletion_performed": False,
        "future_or_sealed_data_read": False, "screening_accessed": False,
        "contract_receipt": contract_receipt, "config_sha256": config_sha,
        "interface_config_sha256": config["interface_config_sha256"], "frets_source_sha256": sha256_file(frets_source),
        "artifacts": {
            "frozen_validation_keys_sha256": sha256_file(expected_path),
            "engineering_receipts_sha256": sha256_file(receipts_path),
            "failure_receipts_sha256": sha256_file(failures_path),
            "unified_predictions_sha256": sha256_file(prediction_path) if prediction_path.is_file() else None,
            **evaluation_hashes,
        },
    }
    metadata["batch_sha256"] = stable_json_sha256(metadata)
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
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
