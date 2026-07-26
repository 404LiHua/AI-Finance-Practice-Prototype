"""Run the two remaining frozen E-5.2 seeds and consolidate the three-seed review."""

from __future__ import annotations

import argparse
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

from stage_e.e5.evaluation import evaluate_predictions, validate_prediction_contract
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve
from stage_e.run_e5_low_cost_single_seed import run as run_single_seed


def verify_existing_batch(root: Path, acceptance_path: Path, base_sha: str, reused_seed: int) -> dict[str, Any]:
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not acceptance["passed"]:
        raise RuntimeError("existing engineering-seed acceptance did not pass")
    if acceptance["config_sha256"] != base_sha or metadata["config_sha256"] != base_sha:
        raise RuntimeError("existing engineering seed uses a different frozen protocol")
    if int(metadata["engineering_seed"]) != reused_seed:
        raise RuntimeError("existing engineering seed differs from the preregistered reused seed")
    required = {
        "frozen_validation_keys_sha256": root / "frozen_validation_keys.csv.gz",
        "engineering_receipts_sha256": root / "engineering_receipts.csv",
        "failure_receipts_sha256": root / "failure_receipts.json",
        "unified_predictions_sha256": root / "unified_predictions.csv.gz",
    }
    for key, path in required.items():
        if sha256_file(path) != metadata["artifacts"][key]:
            raise RuntimeError(f"existing engineering artifact hash mismatch: {key}")
    if not metadata["engineering_pass"] or metadata["failure_count"] != 0:
        raise RuntimeError("existing engineering seed is not a complete passing batch")
    return metadata


def run(config_path: Path, overwrite_new_seeds: bool = False) -> Path:
    review = json.loads(config_path.read_text(encoding="utf-8"))
    if review["status"] != "PREREGISTERED_LOCKED_BEFORE_REMAINING_SEED_RUNS":
        raise ValueError("three-seed review must be locked before remaining seeds run")
    if any(review["restrictions"].values()):
        raise ValueError("three-seed review restrictions must all remain false")
    base_path = resolve(review["base_protocol_config"])
    base_sha = sha256_file(base_path)
    if base_sha != review["base_protocol_config_sha256"]:
        raise RuntimeError("frozen low-cost baseline protocol hash mismatch")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if review["folds"] != base["folds"] or review["seeds"] != base["future_three_seeds"]:
        raise RuntimeError("review folds or seeds differ from the frozen base protocol")
    if review["new_training_seeds"] != [20260723, 20260724] or review["reused_engineering_seed"] != 20260725:
        raise RuntimeError("remaining/reused seed roles differ from preregistration")

    output_root = resolve(review["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    existing_root = resolve(review["paths"]["existing_single_seed_root"])
    existing_acceptance = resolve(review["paths"]["existing_single_seed_acceptance"])
    existing_metadata = verify_existing_batch(
        existing_root, existing_acceptance, base_sha, int(review["reused_engineering_seed"])
    )

    source_roots: dict[int, Path] = {int(review["reused_engineering_seed"]): existing_root}
    for seed in review["new_training_seeds"]:
        seed_root = output_root / "seed_batches" / f"seed_{seed}"
        seed_metadata_path = seed_root / "metadata.json"
        if seed_metadata_path.is_file() and not overwrite_new_seeds:
            seed_metadata = json.loads(seed_metadata_path.read_text(encoding="utf-8"))
            if (
                int(seed_metadata["engineering_seed"]) != int(seed)
                or seed_metadata["config_sha256"] != base_sha
                or not seed_metadata["engineering_pass"]
                or seed_metadata["failure_count"] != 0
            ):
                raise RuntimeError(f"existing new-seed batch cannot be reused: {seed}")
            print(f"seed {seed} verified and reused without retraining", flush=True)
        else:
            run_single_seed(
                base_path, overwrite=overwrite_new_seeds, seed_override=int(seed), output_root_override=seed_root,
            )
        source_roots[int(seed)] = seed_root

    prediction_frames = []
    receipt_frames = []
    expected_reference: pd.DataFrame | None = None
    expected_reference_hash: str | None = None
    source_batches = []
    failure_rows = []
    expected_models = [str(item["id"]) for item in base["models"]]
    for seed in review["seeds"]:
        root = source_roots[int(seed)]
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata["engineering_seed"]) != int(seed) or metadata["config_sha256"] != base_sha:
            raise RuntimeError(f"seed {seed} batch identity mismatch")
        if not metadata["engineering_pass"] or metadata["failure_count"] != 0:
            raise RuntimeError(f"seed {seed} batch failed engineering checks")
        predictions = pd.read_csv(root / "unified_predictions.csv.gz", dtype={"stock_code": str})
        if set(predictions["seed"].astype(int)) != {int(seed)}:
            raise RuntimeError(f"seed {seed} prediction contract contains another seed")
        prediction_frames.append(predictions)
        receipts = pd.read_csv(root / "engineering_receipts.csv")
        receipts.insert(0, "source_batch_root", str(root.relative_to(REPO_ROOT)).replace("\\", "/"))
        receipt_frames.append(receipts)
        failure_rows.extend(json.loads((root / "failure_receipts.json").read_text(encoding="utf-8")))
        expected = pd.read_csv(root / "frozen_validation_keys.csv.gz", dtype={"stock_code": str})
        current_hash = stable_json_sha256(expected.to_dict(orient="records"))
        if expected_reference is None:
            expected_reference, expected_reference_hash = expected, current_hash
        elif current_hash != expected_reference_hash:
            raise RuntimeError(f"seed {seed} frozen validation rows differ")
        source_batches.append({
            "seed": int(seed), "root": str(root.relative_to(REPO_ROOT)).replace("\\", "/"),
            "metadata_sha256": sha256_file(metadata_path), "batch_sha256": metadata["batch_sha256"],
            "reused": int(seed) == int(review["reused_engineering_seed"]),
        })

    assert expected_reference is not None
    predictions = pd.concat(prediction_frames, ignore_index=True)
    receipts = pd.concat(receipt_frames, ignore_index=True)
    predictions_path = output_root / "unified_predictions_three_seed.csv.gz"
    predictions.to_csv(predictions_path, index=False, compression={"method": "gzip", "mtime": 0})
    predictions = pd.read_csv(predictions_path, dtype={"stock_code": str})
    expected_path = output_root / "frozen_validation_keys.csv.gz"
    expected_reference.to_csv(expected_path, index=False, compression={"method": "gzip", "mtime": 0})
    receipts_path = output_root / "engineering_receipts_three_seed.csv"
    receipts.to_csv(receipts_path, index=False)
    failures_path = output_root / "failure_receipts_three_seed.json"
    failures_path.write_text(json.dumps(failure_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    contract = validate_prediction_contract(
        predictions, expected_reference, review["folds"], review["seeds"],
        require_all_frozen_keys=True, allow_extra_rows=False,
    )
    interface = json.loads(resolve(base["interface_config"]).read_text(encoding="utf-8"))
    universe = pd.read_csv(resolve(base["paths"]["universe_path"]), dtype={"stock_code": str})
    evaluations = evaluate_predictions(
        predictions, universe, float(interface["evaluation"]["mape_denominator_floor"]),
        float(interface["evaluation"]["direction_positive_threshold"]),
        int(interface["evaluation"]["return_group_count"]),
    )
    evaluation_hashes = {}
    for name, frame in evaluations.items():
        path = output_root / f"{name}.csv"
        frame.to_csv(path, index=False)
        evaluation_hashes[f"{name}_sha256"] = sha256_file(path)

    metadata = {
        "stage": "E-5.2 three-seed frozen review", "experiment_id": review["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "folds": review["folds"],
        "seeds": review["seeds"], "new_training_seeds": review["new_training_seeds"],
        "reused_engineering_seed": review["reused_engineering_seed"], "models": expected_models,
        "run_count": len(receipts), "failure_count": len(failure_rows),
        "three_seed_review_pass": (
            not failure_rows and len(receipts) == len(expected_models) * len(review["folds"]) * len(review["seeds"])
            and sorted(contract["models"]) == sorted(expected_models)
        ),
        "candidate_selection_performed": False, "model_deletion_performed": False,
        "future_or_sealed_data_read": False, "screening_accessed": False,
        "selection_exposure": review["selection_exposure"], "contract_receipt": contract,
        "review_config_sha256": sha256_file(config_path), "base_protocol_config_sha256": base_sha,
        "existing_engineering_batch_sha256": existing_metadata["batch_sha256"],
        "source_batches": source_batches,
        "artifacts": {
            "unified_predictions_three_seed_sha256": sha256_file(predictions_path),
            "frozen_validation_keys_sha256": sha256_file(expected_path),
            "engineering_receipts_three_seed_sha256": sha256_file(receipts_path),
            "failure_receipts_three_seed_sha256": sha256_file(failures_path),
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
    parser.add_argument("--overwrite-new-seeds", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(config_path, overwrite_new_seeds=args.overwrite_new_seeds))


if __name__ == "__main__":
    main()
