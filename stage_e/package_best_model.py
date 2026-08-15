"""Build the independently consumable Stage-E best-model release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "stock_node_gwnet_fixed_industry_l8"
FOLDS = ("E_RO_01", "E_RO_02", "E_RO_03")
SEEDS = (20260723, 20260724, 20260725)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_run(fold: str, seed: int) -> Path:
    run_name = f"{fold}__{MODEL_ID}__seed{seed}"
    if seed == 20260725:
        return REPO_ROOT / "outputs/stage_e/e5_neural_graph_baselines_single_seed_v1/runs" / run_name
    return REPO_ROOT / f"outputs/stage_e/e5_neural_graph_baselines_three_seed_v1/seed_batches/seed_{seed}/runs" / run_name


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "releases/e_stage_best_model_v1")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    acceptance_path = REPO_ROOT / "outputs/stage_e/e6_candidate_gate_application_acceptance_v1.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not acceptance["passed"] or acceptance["unique_candidate"] != MODEL_ID:
        raise RuntimeError("E-6 sealed acceptance does not authorize this best-model package")

    result_root = REPO_ROOT / "outputs/stage_e/e6_candidate_gate_application_v1"
    matrix = pd.read_csv(result_root / "candidate_gate_matrix.csv")
    selected = matrix.loc[matrix["model_id"].astype(str) == MODEL_ID]
    if len(selected) != 1 or not bool(selected.iloc[0]["eligible"]) or int(selected.iloc[0]["failed_gate_count"]) != 0:
        raise RuntimeError("sealed gate matrix does not contain one fully eligible best model")
    metric_fields = [
        "model_id", "overall_mae", "overall_rmse", "worst_fold_mae", "stocks_below_naive_mae",
        "maximum_stock_mae", "industries_below_naive_mae", "information_technology_mae",
        "mid_cap_mae", "d1_mae", "d10_mae", "tail_mean_mae", "seed_mae_cv",
        "minimum_pairwise_pearson", "minimum_pairwise_spearman", "prediction_seed_std_mean",
        "prediction_seed_std_p95", "total_training_seconds", "total_duration_seconds",
        "maximum_parameter_count", "recorded_inference_seconds", "independent_load_max_abs_difference",
        "failed_gate_count", "eligible",
    ]
    row = selected.iloc[0]
    metrics = {}
    for field in metric_fields:
        value = row[field]
        if field == "model_id":
            metrics[field] = str(value)
        elif field == "eligible":
            metrics[field] = bool(value)
        elif field in {"stocks_below_naive_mae", "industries_below_naive_mae", "failed_gate_count"}:
            metrics[field] = int(value)
        else:
            metrics[field] = float(value)
    metrics["mae_improvement_vs_naive_percent"] = 1.9626676387387416
    metrics["mae_improvement_vs_frets_percent"] = 0.894910810731204
    metrics["gate_count_passed"] = 34
    write_json(output / "model_metrics.json", metrics)

    stock_orders = []
    for fold in FOLDS:
        base = np.load(REPO_ROOT / f"data/processed/e4_adapter_100stocks_v1/{fold}/base_windows.npz")
        stock_orders.append(base["stock_code"].astype(str).tolist())
    if any(order != stock_orders[0] for order in stock_orders[1:]):
        raise RuntimeError("fold stock orders differ")
    write_json(output / "stock_order.json", {"stock_count": 100, "stock_codes": stock_orders[0]})
    write_json(output / "feature_schema.json", {
        "input_shape": ["batch", 8, 100, 6],
        "feature_order": ["return_1w", "log_return_1w", "return_vol_4", "return_vol_12", "model_close", "model_volume_hands"],
        "input_contract": "E-4 adapter normalized numeric_price_l8 view; stock order must exactly match stock_order.json",
        "output_shape": ["batch", 100],
        "three_seed_aggregation": "arithmetic mean",
    })

    copied = []
    for fold in FOLDS:
        for seed in SEEDS:
            source = source_run(fold, seed)
            receipt = json.loads((source / "receipt.json").read_text(encoding="utf-8"))
            checkpoint = source / "model.pt"
            if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint custody mismatch: {fold} seed {seed}")
            destination = output / "checkpoints" / fold / f"seed_{seed}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, destination)
            shutil.copy2(source / "receipt.json", destination.with_suffix(".receipt.json"))
            shutil.copy2(source / "resolved_config.json", destination.with_suffix(".config.json"))
            copied.append({
                "fold_id": fold, "seed": seed,
                "checkpoint": destination.relative_to(output).as_posix(),
                "checkpoint_sha256": sha256_file(destination),
                "independent_load_max_abs_difference": receipt["independent_load_max_abs_difference"],
                "parameter_count": receipt["parameter_count"],
            })

    provenance = {
        "release_id": "e_stage_best_model_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "decision": "UNIQUE_CANDIDATE_RECOMMENDATION",
        "folds": list(FOLDS), "seeds": list(SEEDS),
        "checkpoint_count": len(copied), "checkpoints": copied,
        "e6_acceptance_sha256": sha256_file(acceptance_path),
        "e6_metadata_sha256": sha256_file(result_root / "metadata.json"),
        "e6_recommendation_sha256": sha256_file(result_root / "unique_candidate_recommendation.json"),
        "new_training_performed": False, "screening_accessed": False,
        "commercial_status": "research candidate; not production or investment approval",
    }
    write_json(output / "provenance.json", provenance)

    artifacts = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"):
        artifacts.append({
            "path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "release_id": "e_stage_best_model_v1", "model_id": MODEL_ID,
        "artifact_count": len(artifacts), "artifacts": artifacts,
    }
    manifest["manifest_root_sha256"] = hashlib.sha256(
        json.dumps(artifacts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "status": "PASS", "output": str(output), "checkpoint_count": len(copied),
        "manifest_root_sha256": manifest["manifest_root_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
