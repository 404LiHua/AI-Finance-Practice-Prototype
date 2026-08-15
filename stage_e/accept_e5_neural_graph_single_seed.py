"""Machine acceptance for the frozen E-5.3 single-seed neural/stock-graph run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.e5.evaluation import evaluate_predictions, validate_prediction_contract
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "outputs/stage_e/e5_neural_graph_single_seed_acceptance_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(root / "unified_predictions.csv.gz", dtype={"stock_code": str})
    expected = pd.read_csv(root / "frozen_validation_keys.csv.gz", dtype={"stock_code": str})
    receipts = pd.read_csv(root / "engineering_receipts.csv")
    failures = json.loads((root / "failure_receipts.json").read_text(encoding="utf-8"))
    graph_registry = json.loads((root / "real_stock_graph_registry.json").read_text(encoding="utf-8"))
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    interface = json.loads(resolve(config["interface_config"]).read_text(encoding="utf-8"))
    seed = int(config["engineering_seed"])
    model_ids = [str(item["id"]) for item in config["models"]]
    contract = validate_prediction_contract(predictions, expected, config["folds"], [seed])
    recalculated = evaluate_predictions(
        predictions, universe, float(interface["evaluation"]["mape_denominator_floor"]),
        float(interface["evaluation"]["direction_positive_threshold"]),
        int(interface["evaluation"]["return_group_count"]),
    )
    metric_hashes_valid = True
    metrics_finite = True
    for name, frame in recalculated.items():
        stored = pd.read_csv(root / f"{name}.csv")
        metrics_finite = metrics_finite and bool(np.isfinite(stored.select_dtypes(include=[np.number]).to_numpy()).all())
        rendered = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
        metric_hashes_valid = metric_hashes_valid and rendered == metadata["artifacts"][f"{name}_sha256"]
    payload = dict(metadata)
    declared_batch = payload.pop("batch_sha256", "")
    grid = contract["grid_rows"]
    graph_receipts = receipts.loc[receipts["family"] == "stock_node_gwnet_fixed_industry"]
    timegnn_receipts = receipts.loc[receipts["family"] == "timegnn_stable"]
    checks = {
        "config_locked_before_training": config["status"] == "PREREGISTERED_LOCKED_BEFORE_SINGLE_SEED_RUN",
        "model_set_exactly_four": len(model_ids) == 4 and sorted(contract["models"]) == sorted(model_ids),
        "source_and_attribution_hashes_valid": (
            sha256_file(resolve(config["source_registry"]["implementation"]["path"])) == config["source_registry"]["implementation"]["sha256"]
            and sha256_file(resolve(config["source_registry"]["timegnn_upstream"]["snapshot"])) == config["source_registry"]["timegnn_upstream"]["snapshot_sha256"]
            and sha256_file(resolve(config["source_registry"]["timegnn_upstream"]["attribution"])) == config["source_registry"]["timegnn_upstream"]["attribution_sha256"]
        ),
        "three_folds_one_seed_twelve_receipts": len(receipts) == 12 and metadata["run_count"] == 12,
        "no_failures": not failures and metadata["failure_count"] == 0,
        "prediction_contract_complete": len(grid) == 4 and all(row["row_count"] == 1500 and row["missing_key_count"] == 0 and row["extra_key_count"] == 0 for row in grid),
        "independent_loading_within_tolerance": bool((receipts["independent_load_max_abs_difference"] <= 1e-7).all()),
        "timegnn_is_deterministic_no_gumbel": len(timegnn_receipts) == 3 and bool(timegnn_receipts["deterministic_graph_sampling"].all()),
        "real_stock_graph_has_three_fold_receipts": len(graph_receipts) == 3 and bool(graph_receipts["real_stock_node_graph"].all()),
        "real_stock_graph_registry_valid": len(graph_registry) == 3 and all(row["node_count"] == 100 and row["node_semantics"] == "real_stock_code_nodes" and row["self_loop_count"] == 100 for row in graph_registry),
        "real_stock_graph_order_consistent": len({row["stock_order_sha256"] for row in graph_registry}) == 1 and graph_receipts["stock_order_sha256"].nunique() == 1,
        "metrics_finite": metrics_finite,
        "independent_metric_recalculation_matches": metric_hashes_valid,
        "artifact_hashes_valid": (
            sha256_file(root / "unified_predictions.csv.gz") == metadata["artifacts"]["unified_predictions_sha256"]
            and sha256_file(root / "engineering_receipts.csv") == metadata["artifacts"]["engineering_receipts_sha256"]
            and sha256_file(root / "real_stock_graph_registry.json") == metadata["artifacts"]["real_stock_graph_registry_sha256"]
        ),
        "batch_hash_valid": declared_batch == stable_json_sha256(payload),
        "engineering_pass_declared": bool(metadata["engineering_pass"]),
        "no_selection_deletion_or_promotion": not metadata["candidate_selection_performed"] and not metadata["model_deletion_performed"],
        "future_and_screening_not_accessed": not metadata["future_or_sealed_data_read"] and not metadata["screening_accessed"],
    }
    report = {
        "stage": "E-5.3 neural and real-stock-node graph single-seed acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(checks.values()),
        "checks": checks, "models": model_ids, "folds": config["folds"], "engineering_seed": seed,
        "selection_exposure": config["selection_exposure"], "config_sha256": sha256_file(config_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
