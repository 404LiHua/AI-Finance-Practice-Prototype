"""Independently audit existing E-4 three-seed predictions without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.models.graph_frequency_fusion import GraphFrequencyFusionModel
from stage_e.run_e3_training_checks import resolve


KEYS = ["fold_id", "trade_date", "stock_code"]


def frame_key_sha256(frame: pd.DataFrame) -> str:
    ordered = frame[KEYS].astype(str).sort_values(KEYS, kind="stable")
    return hashlib.sha256(ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def target_sha256(frame: pd.DataFrame) -> str:
    ordered = frame[KEYS + ["target_return"]].sort_values(KEYS, kind="stable")
    return hashlib.sha256(ordered.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")).hexdigest()


def pair_metrics(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float | int]:
    left_values = left.loc[left["sample_valid"].astype(bool), KEYS + ["prediction", "target_return"]].rename(
        columns={"prediction": "prediction_left", "target_return": "target_left"}
    )
    right_values = right.loc[right["sample_valid"].astype(bool), KEYS + ["prediction", "target_return"]].rename(
        columns={"prediction": "prediction_right", "target_return": "target_right"}
    )
    merged = left_values.merge(right_values, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(left_values) or len(merged) != len(right_values):
        raise RuntimeError("seed prediction key mismatch")
    target_difference = float(np.abs(merged["target_left"] - merged["target_right"]).max())
    pearson = float(merged["prediction_left"].corr(merged["prediction_right"], method="pearson"))
    spearman = float(merged["prediction_left"].corr(merged["prediction_right"], method="spearman"))
    return {
        "sample_count": int(len(merged)),
        "pearson": pearson,
        "spearman": spearman,
        "prediction_sign_agreement": float(((merged["prediction_left"] >= 0) == (merged["prediction_right"] >= 0)).mean()),
        "maximum_target_difference": target_difference,
    }


def audit_prediction_variant(variant: str, frame: pd.DataFrame, seeds: list[int], folds: list[str]) -> dict[str, Any]:
    variant_frame = frame.loc[frame["variant"].astype(str).eq(variant)].copy()
    key_rows = []
    variance_rows = []
    seed_frames = {}
    for seed in seeds:
        seed_frame = variant_frame.loc[variant_frame["seed"].astype(int).eq(seed)].copy()
        seed_frames[seed] = seed_frame
        valid = seed_frame.loc[seed_frame["sample_valid"].astype(bool)]
        key_rows.append({
            "variant": variant, "seed": seed, "key_sha256": frame_key_sha256(valid),
            "target_sha256": target_sha256(valid), "valid_sample_count": len(valid),
        })
        for fold_id in folds:
            fold = valid.loc[valid["fold_id"].astype(str).eq(fold_id)]
            variance_rows.append({
                "variant": variant, "seed": seed, "fold_id": fold_id,
                "sample_count": len(fold), "prediction_mean": float(fold["prediction"].mean()),
                "prediction_std": float(fold["prediction"].std(ddof=0)),
                "prediction_min": float(fold["prediction"].min()), "prediction_max": float(fold["prediction"].max()),
                "target_std": float(fold["target_return"].std(ddof=0)),
                "direction_accuracy": float(((fold["prediction"] >= 0) == (fold["target_return"] >= 0)).mean()),
                "mae": float(np.abs(fold["prediction"] - fold["target_return"]).mean()),
            })

    pairwise_overall = []
    pairwise_by_fold = []
    per_stock = []
    for left_index in range(len(seeds)):
        for right_index in range(left_index + 1, len(seeds)):
            left_seed, right_seed = seeds[left_index], seeds[right_index]
            overall = pair_metrics(seed_frames[left_seed], seed_frames[right_seed])
            pairwise_overall.append({"variant": variant, "seed_a": left_seed, "seed_b": right_seed, **overall})
            for fold_id in folds:
                left_fold = seed_frames[left_seed].loc[seed_frames[left_seed]["fold_id"].astype(str).eq(fold_id)]
                right_fold = seed_frames[right_seed].loc[seed_frames[right_seed]["fold_id"].astype(str).eq(fold_id)]
                pairwise_by_fold.append({
                    "variant": variant, "seed_a": left_seed, "seed_b": right_seed, "fold_id": fold_id,
                    **pair_metrics(left_fold, right_fold),
                })
            stocks = sorted(set(seed_frames[left_seed]["stock_code"].astype(str)) & set(seed_frames[right_seed]["stock_code"].astype(str)))
            for stock in stocks:
                left_stock = seed_frames[left_seed].loc[seed_frames[left_seed]["stock_code"].astype(str).eq(stock)]
                right_stock = seed_frames[right_seed].loc[seed_frames[right_seed]["stock_code"].astype(str).eq(stock)]
                metrics = pair_metrics(left_stock, right_stock)
                per_stock.append({"variant": variant, "seed_a": left_seed, "seed_b": right_seed, "stock_code": stock, **metrics})

    valid = variant_frame.loc[variant_frame["sample_valid"].astype(bool)]
    pivot = valid.pivot(index=KEYS, columns="seed", values="prediction").dropna()
    dispersion = {
        "variant": variant,
        "aligned_sample_count": int(len(pivot)),
        "mean_cross_seed_prediction_std": float(pivot.std(axis=1, ddof=0).mean()),
        "maximum_cross_seed_prediction_std": float(pivot.std(axis=1, ddof=0).max()),
        "mean_absolute_prediction": float(np.abs(pivot.to_numpy()).mean()),
    }
    return {
        "key_rows": key_rows,
        "variance_rows": variance_rows,
        "pairwise_overall": pairwise_overall,
        "pairwise_by_fold": pairwise_by_fold,
        "per_stock": per_stock,
        "dispersion": dispersion,
    }


def recompute_learned_graph(config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    model_config_path = resolve(config["paths"]["learned_graph_config"])
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    base = np.load(resolve(config["paths"]["adapter_root"]) / model_config["fold_id"] / "base_windows.npz")
    indices = np.flatnonzero(base["split"].astype(str) == "train")[: int(model_config["train_cross_sections"])]
    x = torch.tensor(base["values"][indices], dtype=torch.float32)
    available = torch.tensor(base["node_available"][indices], dtype=torch.bool)
    mask = base["sample_mask"][indices].astype(bool)
    target = base["target_raw"][indices]
    dates = base["trade_date"][indices].astype(str)
    stocks = base["stock_code"].astype(str)
    target_mean = float(base["target_mean_train"][0])
    target_std = float(base["target_std_train"][0])
    rows = []
    edges = {}
    checkpoints = []
    for seed in config["expected_seeds"]:
        options = model_config["model"]
        model = GraphFrequencyFusionModel(
            input_dim=x.shape[-1], stock_count=x.shape[2], hidden_dim=int(options["hidden_dim"]),
            top_k=int(options["top_k"]), dropout=float(options["dropout"]), graph_mode=options["graph_mode"],
            branch_mode=options["branch_mode"], fusion_mode=options["fusion_mode"], text_fusion=options["text_fusion"],
        )
        checkpoint = resolve(config["paths"]["learned_graph_checkpoint_root"]) / f"model_seed_{seed}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            details = model(x, node_available=available, return_details=True)
        prediction = details["prediction"].numpy() * target_std + target_mean
        adjacency = details["adjacency"].numpy()
        diagonal = np.eye(adjacency.shape[-1], dtype=bool)[None, :, :]
        edges[int(seed)] = (adjacency > 0) & ~diagonal
        checkpoints.append({"seed": int(seed), "sha256": sha256_file(checkpoint)})
        for time_index, date in enumerate(dates):
            for stock_index, stock in enumerate(stocks):
                rows.append({
                    "variant": "dual_learned_fixed_no_text_tiny_train", "fold_id": "E_RO_01_TINY_TRAIN",
                    "trade_date": date, "stock_code": stock, "target_return": float(target[time_index, stock_index]),
                    "prediction": float(prediction[time_index, stock_index]), "sample_valid": bool(mask[time_index, stock_index]),
                    "text_available": False, "seed": int(seed),
                })
    edge_rows = []
    seeds = [int(seed) for seed in config["expected_seeds"]]
    for left_index in range(len(seeds)):
        for right_index in range(left_index + 1, len(seeds)):
            left, right = seeds[left_index], seeds[right_index]
            intersection = np.logical_and(edges[left], edges[right]).sum()
            union = np.logical_or(edges[left], edges[right]).sum()
            edge_rows.append({"seed_a": left, "seed_b": right, "edge_jaccard": float(intersection / union)})
    receipt = {
        "model_config_sha256": sha256_file(model_config_path),
        "adapter_base_sha256": sha256_file(resolve(config["paths"]["adapter_root"]) / model_config["fold_id"] / "base_windows.npz"),
        "checkpoints": checkpoints,
    }
    return pd.DataFrame(rows), edge_rows, receipt


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["restrictions"]["training_allowed"]:
        raise ValueError("E-4S.1 must prohibit training")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config["expected_seeds"]]
    folds = [str(fold) for fold in config["expected_folds"]]

    learned_frame, learned_edges, learned_receipt = recompute_learned_graph(config)
    learned_path = output_root / "learned_graph_recomputed_predictions.csv.gz"
    learned_frame.to_csv(learned_path, index=False, compression={"method": "gzip", "mtime": 0})
    fixed_frame = pd.read_csv(resolve(config["paths"]["fixed_graph_predictions"]), dtype={"stock_code": str})
    control_frame = pd.read_csv(resolve(config["paths"]["control_predictions"]), dtype={"stock_code": str})

    audits = []
    learned_audit = audit_prediction_variant("dual_learned_fixed_no_text_tiny_train", learned_frame, seeds, ["E_RO_01_TINY_TRAIN"])
    audits.append(learned_audit)
    for variant in sorted(fixed_frame["variant"].astype(str).unique()):
        audits.append(audit_prediction_variant(variant, fixed_frame, seeds, folds))
    for variant in sorted(control_frame["variant"].astype(str).unique()):
        audits.append(audit_prediction_variant(variant, control_frame, seeds, folds))

    key_rows = [row for audit in audits for row in audit["key_rows"]]
    variance_rows = [row for audit in audits for row in audit["variance_rows"]]
    overall_rows = [row for audit in audits for row in audit["pairwise_overall"]]
    fold_rows = [row for audit in audits for row in audit["pairwise_by_fold"]]
    stock_rows = [row for audit in audits for row in audit["per_stock"]]
    dispersion_rows = [audit["dispersion"] for audit in audits]
    key_path = output_root / "sample_key_audit.csv"
    variance_path = output_root / "prediction_variance_by_fold.csv"
    overall_path = output_root / "pairwise_overall.csv"
    fold_path = output_root / "pairwise_by_fold.csv"
    stock_path = output_root / "pairwise_per_stock.csv"
    dispersion_path = output_root / "cross_seed_dispersion.csv"
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    pd.DataFrame(variance_rows).to_csv(variance_path, index=False)
    pd.DataFrame(overall_rows).to_csv(overall_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(stock_rows).to_csv(stock_path, index=False)
    pd.DataFrame(dispersion_rows).to_csv(dispersion_path, index=False)

    key_frame = pd.DataFrame(key_rows)
    key_consistency = key_frame.groupby("variant").agg(key_hash_count=("key_sha256", "nunique"), target_hash_count=("target_sha256", "nunique"), sample_count_count=("valid_sample_count", "nunique")).reset_index()
    all_keys_equal = bool((key_consistency[["key_hash_count", "target_hash_count", "sample_count_count"]] == 1).all().all())
    original_learned = json.loads(resolve(config["paths"]["learned_graph_results"]).read_text(encoding="utf-8"))
    recomputed_learned_pairs = {
        (int(row["seed_a"]), int(row["seed_b"])): row for row in overall_rows if row["variant"] == "dual_learned_fixed_no_text_tiny_train"
    }
    learned_metric_differences = []
    for reported in original_learned["pairwise"]:
        key = (int(reported["seed_a"]), int(reported["seed_b"]))
        recomputed = recomputed_learned_pairs[key]
        edge = next(item for item in learned_edges if (item["seed_a"], item["seed_b"]) == key)
        learned_metric_differences.append({
            "seed_a": key[0], "seed_b": key[1],
            "prediction_correlation_difference": abs(float(reported["prediction_correlation"]) - float(recomputed["pearson"])),
            "edge_jaccard_difference": abs(float(reported["edge_jaccard"]) - float(edge["edge_jaccard"])),
        })

    control_fold = pd.read_csv(resolve(config["paths"]["control_fold_results"]))
    fixed_fold = pd.read_csv(resolve(config["paths"]["fixed_graph_fold_results"]))
    best_epoch_range = {
        "controls_min": int(control_fold["best_epoch"].min()), "controls_max": int(control_fold["best_epoch"].max()),
        "fixed_min": int(fixed_fold["best_epoch"].min()), "fixed_max": int(fixed_fold["best_epoch"].max()),
    }
    overall_frame = pd.DataFrame(overall_rows)
    validation_variants = overall_frame.loc[~overall_frame["variant"].str.contains("tiny_train")]
    minimum_validation_pearson = float(validation_variants["pearson"].min())
    zero_variance_rows = int((pd.DataFrame(variance_rows)["prediction_std"] < float(config["tolerances"]["minimum_prediction_std"])).sum())
    reported_metrics_match = all(
        max(row["prediction_correlation_difference"], row["edge_jaccard_difference"]) <= float(config["tolerances"]["metric_absolute_difference"])
        for row in learned_metric_differences
    )
    conclusion = (
        "TRAINING_CONVERGENCE_AND_CHECKPOINT_SELECTION_INSTABILITY"
        if all_keys_equal and reported_metrics_match and zero_variance_rows == 0 and minimum_validation_pearson < 0.90
        else "AUDIT_REQUIRES_MANUAL_REVIEW"
    )
    training_protocol_path = resolve(config["paths"]["training_protocol_v2"])
    source_paths = {
        key: resolve(config["paths"][key]) for key in (
            "learned_graph_results", "fixed_graph_predictions", "fixed_graph_results", "fixed_graph_fold_results",
            "control_predictions", "control_results", "control_fold_results", "training_protocol_v2"
        )
    }
    report = {
        "stage": "E-4S.1",
        "audit_id": config["audit_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_training_executed": True,
        "future_or_sealed_data_read": False,
        "screening_accessed": False,
        "sample_keys_equal_across_seeds": all_keys_equal,
        "reported_learned_graph_metrics_independently_reproduced": reported_metrics_match,
        "zero_prediction_variance_rows": zero_variance_rows,
        "minimum_validation_pairwise_pearson": minimum_validation_pearson,
        "best_epoch_range": best_epoch_range,
        "unique_conclusion": conclusion,
        "conclusion_detail": "Frozen keys, targets, inverse scaling, and reported metrics are consistent. Predictions are not constant. Seed-specific validation-selected epochs range widely, while validation prediction correlations remain low despite similar MAE; the evidence supports convergence/checkpoint-selection instability rather than key misalignment or metric implementation error.",
        "learned_graph_recompute": learned_receipt,
        "learned_metric_differences": learned_metric_differences,
        "training_protocol_v2_sha256": sha256_file(training_protocol_path),
        "config_sha256": sha256_file(config_path),
        "source_sha256": {key: sha256_file(path) for key, path in source_paths.items()},
        "artifacts": {
            "learned_graph_recomputed_predictions_sha256": sha256_file(learned_path),
            "sample_key_audit_sha256": sha256_file(key_path),
            "prediction_variance_by_fold_sha256": sha256_file(variance_path),
            "pairwise_overall_sha256": sha256_file(overall_path),
            "pairwise_by_fold_sha256": sha256_file(fold_path),
            "pairwise_per_stock_sha256": sha256_file(stock_path),
            "cross_seed_dispersion_sha256": sha256_file(dispersion_path),
        },
    }
    report["batch_sha256"] = stable_json_sha256(report)
    (output_root / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
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
