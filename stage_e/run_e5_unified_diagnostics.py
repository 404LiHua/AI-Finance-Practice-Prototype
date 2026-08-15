"""Run frozen E-5.4 diagnostics by merging existing E-5.2 and E-5.3 predictions only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.e5.diagnostics import (
    engineering_cost_summary, fold_and_worst_fold, model_disagreement,
    overall_pooled_metrics, seed_prediction_dispersion,
)
from stage_e.e5.evaluation import evaluate_predictions, validate_prediction_contract
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "DIAGNOSTIC_RULES_FROZEN_BEFORE_CROSS_BATCH_READ":
        raise ValueError("E-5.4 diagnostic rules must be frozen before reading both batches")
    if any(config["restrictions"].values()):
        raise ValueError("all E-5.4 mutation permissions must remain false")
    interface_path = resolve(config["interface_config"])
    if sha256_file(interface_path) != config["interface_config_sha256"]:
        raise RuntimeError("E-5 interface hash mismatch")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    receipt_frames = []
    source_registry = []
    frozen_keys_reference: pd.DataFrame | None = None
    frozen_keys_hash: str | None = None
    for source in config["sources"]:
        for field, hash_field in (
            ("metadata", "metadata_sha256"), ("predictions", "predictions_sha256"),
            ("receipts", "receipts_sha256"), ("acceptance", "acceptance_sha256"),
        ):
            if sha256_file(resolve(source[field])) != source[hash_field]:
                raise RuntimeError(f"source hash mismatch: {source['source_id']} {field}")
        metadata = json.loads(resolve(source["metadata"]).read_text(encoding="utf-8"))
        acceptance = json.loads(resolve(source["acceptance"]).read_text(encoding="utf-8"))
        if metadata["batch_sha256"] != source["batch_sha256"] or not acceptance["passed"]:
            raise RuntimeError(f"source batch not accepted: {source['source_id']}")
        predictions = pd.read_csv(resolve(source["predictions"]), dtype={"stock_code": str})
        receipts = pd.read_csv(resolve(source["receipts"]), dtype={"stock_code": str})
        predictions["source_id"] = source["source_id"]
        receipts["diagnostic_source_id"] = source["source_id"]
        prediction_frames.append(predictions)
        receipt_frames.append(receipts)
        frozen_keys = pd.read_csv(resolve(source["frozen_keys"]), dtype={"stock_code": str})
        current_hash = stable_json_sha256(frozen_keys.to_dict(orient="records"))
        if frozen_keys_reference is None:
            frozen_keys_reference, frozen_keys_hash = frozen_keys, current_hash
        elif current_hash != frozen_keys_hash:
            raise RuntimeError("E-5.2 and E-5.3 frozen validation rows differ")
        source_registry.append({
            "source_id": source["source_id"], "batch_sha256": source["batch_sha256"],
            "metadata_sha256": source["metadata_sha256"], "predictions_sha256": source["predictions_sha256"],
            "receipts_sha256": source["receipts_sha256"], "acceptance_sha256": source["acceptance_sha256"],
            "model_count": int(predictions["model_id"].nunique()), "prediction_rows": int(len(predictions)),
        })
    assert frozen_keys_reference is not None

    predictions = pd.concat(prediction_frames, ignore_index=True)
    receipts = pd.concat(receipt_frames, ignore_index=True)
    model_ids = [str(item["id"]) for item in config["models"]]
    if sorted(predictions["model_id"].astype(str).unique()) != sorted(model_ids):
        raise RuntimeError("merged prediction model set differs from frozen ten")
    if predictions.duplicated(["model_id", "seed", "fold_id", "sample_row_id"]).any():
        raise RuntimeError("cross-batch merge creates duplicate model prediction keys")
    prediction_path = output_root / "unified_predictions_ten_models.csv.gz"
    predictions.to_csv(prediction_path, index=False, compression={"method": "gzip", "mtime": 0})
    predictions = pd.read_csv(prediction_path, dtype={"stock_code": str})
    receipts_path = output_root / "engineering_receipts_ten_models.csv"
    receipts.to_csv(receipts_path, index=False)
    frozen_keys_path = output_root / "frozen_validation_keys.csv.gz"
    frozen_keys_reference.to_csv(frozen_keys_path, index=False, compression={"method": "gzip", "mtime": 0})
    contract = validate_prediction_contract(predictions, frozen_keys_reference, config["folds"], config["seeds"])

    interface = json.loads(interface_path.read_text(encoding="utf-8"))
    mape_floor = float(interface["evaluation"]["mape_denominator_floor"])
    direction_threshold = float(config["diagnostic_rules"]["direction_positive_threshold"])
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    standard = evaluate_predictions(
        predictions, universe, mape_floor, direction_threshold,
        int(config["diagnostic_rules"]["return_group_count"]),
    )
    diagnostics = dict(standard)
    diagnostics["overall_pooled_metrics"] = overall_pooled_metrics(predictions, mape_floor, direction_threshold)
    diagnostics["fold_pooled_metrics"], diagnostics["worst_fold_summary"] = fold_and_worst_fold(
        predictions, mape_floor, direction_threshold,
    )
    diagnostics["engineering_cost_summary"] = engineering_cost_summary(receipts)
    diagnostics["seed_prediction_dispersion"] = seed_prediction_dispersion(predictions)
    component_map = {str(item["id"]): str(item["component_family"]) for item in config["models"]}
    diagnostics["model_disagreement"], diagnostics["component_disagreement"] = model_disagreement(
        predictions, component_map, direction_threshold,
    )

    artifact_hashes = {
        "unified_predictions_ten_models_sha256": sha256_file(prediction_path),
        "engineering_receipts_ten_models_sha256": sha256_file(receipts_path),
        "frozen_validation_keys_sha256": sha256_file(frozen_keys_path),
    }
    for name, frame in diagnostics.items():
        path = output_root / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifact_hashes[f"{name}_sha256"] = sha256_file(path)

    finite = True
    for frame in diagnostics.values():
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        finite = finite and bool(np.isfinite(numeric).all())
    metadata = {
        "stage": "E-5.4 frozen unified diagnostics", "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "models": model_ids,
        "folds": config["folds"], "seeds": config["seeds"], "prediction_rows": int(len(predictions)),
        "receipt_rows": int(len(receipts)), "contract_receipt": contract,
        "diagnostics_finite": finite, "training_executed": False, "checkpoint_written": False,
        "candidate_ranking_performed": False, "candidate_selection_performed": False,
        "model_deletion_performed": False, "future_or_sealed_data_read": False, "screening_accessed": False,
        "selection_exposure": config["selection_exposure"], "config_sha256": sha256_file(config_path),
        "interface_config_sha256": config["interface_config_sha256"], "source_registry": source_registry,
        "diagnostic_rule_sha256": stable_json_sha256(config["diagnostic_rules"]), "artifacts": artifact_hashes,
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
