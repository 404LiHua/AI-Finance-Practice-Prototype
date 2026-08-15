"""Run the preregistered E-5.3 neural and real-stock-node graph baselines for one seed."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.e5.evaluation import evaluate_predictions, validate_prediction_contract
from stage_e.e5.interface import load_fold_view, validation_key_frame
from stage_e.e5.neural_graph import fixed_industry_adjacency, load_predict, train_model
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
        raise ValueError("E-5.3 model set must be frozen before training")
    if config["restrictions"]["candidate_selection_allowed"] or config["restrictions"]["model_deletion_allowed"]:
        raise ValueError("single-seed engineering cannot select or delete models")
    interface_path = resolve(config["interface_config"])
    if sha256_file(interface_path) != config["interface_config_sha256"]:
        raise RuntimeError("frozen E-5 interface config hash mismatch")
    implementation_path = resolve(config["source_registry"]["implementation"]["path"])
    if sha256_file(implementation_path) != config["source_registry"]["implementation"]["sha256"]:
        raise RuntimeError("frozen E-5.3 implementation hash mismatch")
    upstream = config["source_registry"]["timegnn_upstream"]
    if sha256_file(resolve(upstream["snapshot"])) != upstream["snapshot_sha256"]:
        raise RuntimeError("Time-GNN upstream snapshot hash mismatch")
    if sha256_file(resolve(upstream["attribution"])) != upstream["attribution_sha256"]:
        raise RuntimeError("Time-GNN attribution hash mismatch")

    output_root = output_root_override or resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_root = resolve(config["paths"]["adapter_root"])
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    interface = json.loads(interface_path.read_text(encoding="utf-8"))
    config_sha = sha256_file(config_path)
    seed = int(config["engineering_seed"] if seed_override is None else seed_override)
    if seed not in {int(value) for value in config["future_three_seeds"]}:
        raise ValueError(f"seed override is outside the preregistered set: {seed}")
    all_predictions = []
    receipts = []
    failures = []
    expected_keys = []
    fold_graph_registry = []

    for fold_id in config["folds"]:
        view = load_fold_view(adapter_root, fold_id, "no_text")
        key_frame = validation_key_frame(view)
        expected_keys.append(key_frame)
        adjacency, industries = fixed_industry_adjacency(view.stock_code, universe)
        stock_order_sha = stable_json_sha256(view.stock_code.astype(str).tolist())
        adjacency_sha = stable_json_sha256(adjacency.tolist())
        if adjacency.shape != (100, 100) or len(set(view.stock_code.astype(str))) != 100:
            raise RuntimeError(f"{fold_id} real-stock-node graph is not the frozen 100-stock order")
        fold_graph_registry.append({
            "fold_id": fold_id, "node_count": 100, "node_semantics": "real_stock_code_nodes",
            "stock_order_sha256": stock_order_sha, "adjacency_sha256": adjacency_sha,
            "industry_count": len(set(industries)), "self_loop_count": int(np.diag(adjacency > 0).sum()),
        })
        for model_spec in config["models"]:
            model_id = str(model_spec["id"])
            family = str(model_spec["family"])
            parameters = dict(model_spec["parameters"])
            run_dir = output_root / "runs" / f"{fold_id}__{model_id}__seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            resolved_path = run_dir / "resolved_config.json"
            resolved_payload = {
                "experiment_id": config["experiment_id"], "model_id": model_id, "family": family,
                "feature_view": model_spec["feature_view"], "node_semantics": model_spec["node_semantics"],
                "parameters": parameters, "fold_id": fold_id, "seed": seed,
                "interface_config_sha256": config["interface_config_sha256"],
                "implementation_sha256": config["source_registry"]["implementation"]["sha256"],
                "stock_order_sha256": stock_order_sha,
                "adjacency_sha256": adjacency_sha if family == "stock_node_gwnet_fixed_industry" else None,
                "selection_exposure": config["selection_exposure"], "future_or_sealed_data_read": False,
            }
            resolved_path.write_text(json.dumps(resolved_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            checkpoint = run_dir / "model.pt"
            started = time.perf_counter()
            try:
                model_adjacency = adjacency if family == "stock_node_gwnet_fixed_industry" else None
                prediction, detail = train_model(
                    family, view, parameters, seed, model_adjacency, checkpoint, run_dir / "training_log.json",
                )
                loaded = load_predict(checkpoint, view)
                load_difference = float(np.max(np.abs(prediction - loaded)))
                if load_difference > float(config["independent_loading"]["maximum_prediction_absolute_difference"]):
                    raise RuntimeError(f"independent load prediction mismatch: {load_difference}")
                if not np.isfinite(prediction).all():
                    raise RuntimeError("non-finite E-5.3 prediction")
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
                    "node_semantics": model_spec["node_semantics"],
                    "deterministic_graph_sampling": family != "timegnn_stable" or model_spec["stabilization"].startswith("deterministic"),
                    "real_stock_node_graph": family == "stock_node_gwnet_fixed_industry",
                    "stock_order_sha256": stock_order_sha,
                    "adjacency_sha256": adjacency_sha if family == "stock_node_gwnet_fixed_industry" else None,
                    "independent_load_max_abs_difference": load_difference,
                    "checkpoint_sha256": checkpoint_sha, "resolved_config_sha256": sha256_file(resolved_path),
                    "duration_seconds": time.perf_counter() - started, **detail,
                }
                receipts.append(receipt)
                (run_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"{fold_id} {model_id} PASS rows={len(rows)} load_diff={load_difference:.3g}", flush=True)
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
    graph_registry_path = output_root / "real_stock_graph_registry.json"
    graph_registry_path.write_text(json.dumps(fold_graph_registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions_path = output_root / "unified_predictions.csv.gz"
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    if not predictions.empty:
        predictions.to_csv(predictions_path, index=False, compression={"method": "gzip", "mtime": 0})
        predictions = pd.read_csv(predictions_path, dtype={"stock_code": str})

    contract = None
    evaluation_hashes = {}
    completed_models = []
    if not predictions.empty:
        contract = validate_prediction_contract(predictions, expected, config["folds"], [seed])
        completed_models = sorted(predictions["model_id"].astype(str).unique())
        evaluations = evaluate_predictions(
            predictions, universe, float(interface["evaluation"]["mape_denominator_floor"]),
            float(interface["evaluation"]["direction_positive_threshold"]),
            int(interface["evaluation"]["return_group_count"]),
        )
        for name, frame in evaluations.items():
            path = output_root / f"{name}.csv"
            frame.to_csv(path, index=False)
            evaluation_hashes[f"{name}_sha256"] = sha256_file(path)
    expected_models = [str(item["id"]) for item in config["models"]]
    engineering_pass = not failures and completed_models == sorted(expected_models) and len(receipts) == 12
    metadata = {
        "stage": "E-5.3 neural and real-stock-node graph single-seed engineering",
        "experiment_id": config["experiment_id"], "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "engineering_seed": seed, "folds": config["folds"], "expected_models": expected_models,
        "completed_models": completed_models, "run_count": len(receipts), "failure_count": len(failures),
        "engineering_pass": engineering_pass, "candidate_selection_performed": False,
        "model_deletion_performed": False, "future_or_sealed_data_read": False, "screening_accessed": False,
        "contract_receipt": contract, "config_sha256": config_sha,
        "interface_config_sha256": config["interface_config_sha256"],
        "implementation_sha256": config["source_registry"]["implementation"]["sha256"],
        "timegnn_upstream_snapshot_sha256": upstream["snapshot_sha256"],
        "fold_graph_registry": fold_graph_registry,
        "artifacts": {
            "frozen_validation_keys_sha256": sha256_file(expected_path),
            "engineering_receipts_sha256": sha256_file(receipts_path),
            "failure_receipts_sha256": sha256_file(failures_path),
            "real_stock_graph_registry_sha256": sha256_file(graph_registry_path),
            "unified_predictions_sha256": sha256_file(predictions_path) if predictions_path.is_file() else None,
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
