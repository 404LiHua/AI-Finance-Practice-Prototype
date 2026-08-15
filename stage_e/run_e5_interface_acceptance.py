"""Build and evaluate a no-training E-5 interface acceptance fixture."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.e5.evaluation import evaluate_predictions, validate_prediction_contract
from stage_e.e5.interface import load_fold_view, validation_key_frame
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["restrictions"]["baseline_training_allowed"]:
        raise ValueError("E-5.1 acceptance must not train baselines")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_root = resolve(config["paths"]["adapter_root"])

    expected_frames = []
    text_view_hashes = {}
    for fold_id in config["folds"]:
        base_view = load_fold_view(adapter_root, fold_id, "no_text")
        expected_frames.append(validation_key_frame(base_view))
        base_ids = base_view.sample_row_id
        for text_view in config["required_text_views"]:
            view = load_fold_view(adapter_root, fold_id, text_view)
            if not np.array_equal(base_ids, view.sample_row_id):
                raise RuntimeError(f"E-5 text view key mismatch: {fold_id} {text_view}")
            text_view_hashes[f"{fold_id}/{text_view}"] = sha256_file(adapter_root / fold_id / f"text_{text_view}.npz")
    expected_keys = pd.concat(expected_frames, ignore_index=True)
    expected_keys_path = output_root / "frozen_validation_keys.csv.gz"
    expected_keys.to_csv(expected_keys_path, index=False, compression={"method": "gzip", "mtime": 0})

    fixture = pd.read_csv(resolve(config["paths"]["acceptance_fixture_predictions"]), dtype={"stock_code": str})
    fold_results = pd.read_csv(resolve(config["paths"]["acceptance_fixture_fold_results"]))
    fixture_config_path = resolve(config["paths"]["acceptance_fixture_config"])
    fixture_config_sha = sha256_file(fixture_config_path)
    checkpoints = fold_results[["variant", "fold_id", "seed", "checkpoint_sha256"]].copy()
    fixture = fixture.merge(checkpoints, on=["variant", "fold_id", "seed"], how="left", validate="many_to_one")
    if fixture["checkpoint_sha256"].isna().any():
        raise RuntimeError("acceptance fixture checkpoint hash is incomplete")
    fixture = fixture.merge(
        expected_keys[["fold_id", "trade_date", "stock_code", "sample_row_id", "target_date", "target_return"]].rename(columns={"target_return": "expected_target_return"}),
        on=["fold_id", "trade_date", "stock_code"], how="left", validate="many_to_one",
    )
    if fixture["sample_row_id"].isna().any() or float(np.abs(fixture["target_return"] - fixture["expected_target_return"]).max()) > 1e-7:
        raise RuntimeError("acceptance fixture does not align with frozen E-5 validation keys")
    fixture = fixture.rename(columns={"variant": "model_id"})
    fixture["config_sha256"] = fixture_config_sha
    contract_columns = [
        "model_id", "seed", "fold_id", "sample_row_id", "trade_date", "target_date", "stock_code",
        "target_return", "prediction", "sample_valid", "text_available", "checkpoint_sha256", "config_sha256",
    ]
    unified = fixture[contract_columns].copy()
    contract_receipt = validate_prediction_contract(
        unified, expected_keys, config["folds"], config["seeds"],
        require_all_frozen_keys=bool(config["evaluation"]["require_all_frozen_keys"]),
        allow_extra_rows=bool(config["evaluation"]["allow_extra_prediction_rows"]),
    )
    predictions_path = output_root / "unified_predictions.csv.gz"
    unified.to_csv(predictions_path, index=False, compression={"method": "gzip", "mtime": 0})

    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    evaluations = evaluate_predictions(
        unified, universe, float(config["evaluation"]["mape_denominator_floor"]),
        float(config["evaluation"]["direction_positive_threshold"]), int(config["evaluation"]["return_group_count"]),
    )
    artifact_hashes = {}
    for name, frame in evaluations.items():
        path = output_root / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifact_hashes[f"{name}_sha256"] = sha256_file(path)

    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "executable": sys.executable,
    }
    environment_path = output_root / "environment.json"
    environment_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "stage": "E-5.1", "interface_id": config["interface_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_training_executed": False, "future_or_sealed_data_read": False, "screening_accessed": False,
        "folds": config["folds"], "seeds": config["seeds"], "models": contract_receipt["models"],
        "prediction_contract_grid": contract_receipt["grid_rows"],
        "config_sha256": sha256_file(config_path), "prediction_schema_sha256": sha256_file(resolve(config["paths"]["prediction_schema"])),
        "fixture_config_sha256": fixture_config_sha, "text_view_hashes": text_view_hashes,
        "artifacts": {
            "frozen_validation_keys_sha256": sha256_file(expected_keys_path),
            "unified_predictions_sha256": sha256_file(predictions_path),
            "environment_sha256": sha256_file(environment_path), **artifact_hashes,
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
