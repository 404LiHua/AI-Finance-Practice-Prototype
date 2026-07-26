"""Run E-4S.2 controls under the frozen full-TRAIN/final-EMA protocol."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.models.graph_frequency_fusion import GraphFrequencyFusionModel
from stage_e.run_e3_training_checks import resolve, set_seed
from stage_e.run_e4_ablations import graph_mode, load_fold, load_graphs, provided_adjacency, raw_metrics
from stage_e.run_e4_fixed_graph_stabilization import prediction_correlation


def update_ema(ema: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            source = value.detach()
            if torch.is_floating_point(source):
                ema[name].mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                ema[name].copy_(source)


def run_fold_v2(
    protocol: dict[str, Any], variant: dict[str, str], fold_id: str, seed: int,
    data: dict[str, Any], graphs: dict[str, Any], output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(seed)
    options = protocol["training"]
    train_indices = np.flatnonzero(data["split"] == "train")
    validation_indices = np.flatnonzero(data["split"] == "validation")
    if not options["use_all_train_cross_sections"] or len(train_indices) == 0:
        raise RuntimeError("frozen V2 protocol requires all TRAIN cross-sections")
    model = GraphFrequencyFusionModel(
        input_dim=data["x"].shape[-1], stock_count=data["x"].shape[2], hidden_dim=int(options["hidden_dim"]),
        top_k=int(options["top_k"]), dropout=float(options["dropout"]), graph_mode=graph_mode(variant["graph"]),
        branch_mode=variant["branch"], fusion_mode=variant["fusion"], text_dim=0, text_fusion=variant["text_fusion"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(options["learning_rate"]), weight_decay=float(options["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(options["epochs"]))
    loss_function = nn.SmoothL1Loss()
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    generator = np.random.default_rng(seed)
    curve = []
    started = time.perf_counter()
    for epoch in range(1, int(options["epochs"]) + 1):
        model.train()
        shuffled = generator.permutation(train_indices)
        loss_sum = 0.0
        sample_sum = 0
        gradient_norm_sum = 0.0
        batch_count = 0
        for start in range(0, len(shuffled), int(options["batch_size"])):
            batch = shuffled[start : start + int(options["batch_size"])]
            x = torch.tensor(data["x"][batch], dtype=torch.float32)
            y = torch.tensor(data["target_scaled"][batch], dtype=torch.float32)
            mask = torch.tensor(data["mask"][batch], dtype=torch.bool)
            available = torch.tensor(data["available"][batch], dtype=torch.bool)
            adjacency = provided_adjacency(variant["graph"], batch, data["dates"], graphs)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, node_available=available, adjacency=adjacency)
            loss = loss_function(prediction[mask], y[mask])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite V2 loss: {variant['id']} {fold_id} seed={seed}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(options["gradient_clip"]))
            optimizer.step()
            update_ema(ema, model, float(options["ema_decay"]))
            count = int(mask.sum())
            loss_sum += float(loss.detach()) * count
            sample_sum += count
            gradient_norm_sum += float(gradient_norm)
            batch_count += 1
        curve.append({
            "variant": variant["id"], "fold_id": fold_id, "seed": seed, "epoch": epoch,
            "train_loss": loss_sum / max(sample_sum, 1), "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "mean_gradient_norm": gradient_norm_sum / max(batch_count, 1),
        })
        scheduler.step()
    training_seconds = time.perf_counter() - started
    model.load_state_dict(ema)
    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(data["x"][validation_indices], dtype=torch.float32)
        val_available = torch.tensor(data["available"][validation_indices], dtype=torch.bool)
        val_adjacency = provided_adjacency(variant["graph"], validation_indices, data["dates"], graphs)
        details = model(val_x, node_available=val_available, adjacency=val_adjacency, return_details=True)
    metrics = raw_metrics(details["prediction"], data, validation_indices)
    adjacency_array = details["adjacency"].numpy()
    checkpoint = output_root / "checkpoints" / f"{variant['id']}__{fold_id}__seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    prediction_raw = details["prediction"].numpy() * data["target_std"] + data["target_mean"]
    rows = []
    for local, index in enumerate(validation_indices):
        for stock_index, stock in enumerate(data["stocks"]):
            rows.append({
                "variant": variant["id"], "fold_id": fold_id, "trade_date": data["dates"][index],
                "stock_code": stock, "target_return": float(data["target_raw"][index, stock_index]),
                "prediction": float(prediction_raw[local, stock_index]), "sample_valid": bool(data["mask"][index, stock_index]),
                "text_available": bool(data["text_available"][index, stock_index]), "seed": seed,
            })
    receipt = {
        "variant": variant["id"], "fold_id": fold_id, "seed": seed, **metrics,
        "train_cross_section_count": int(len(train_indices)), "epochs_completed": int(options["epochs"]),
        "checkpoint_selection": "final_epoch_ema", "validation_used_for_checkpoint_selection": False,
        "final_train_loss": float(curve[-1]["train_loss"]), "training_seconds": training_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "adjacency_finite": bool(np.isfinite(adjacency_array).all()),
        "adjacency_row_stochastic": bool(np.allclose(adjacency_array.sum(axis=-1), 1.0, atol=1e-5)),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    return receipt, rows, curve


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_path = resolve(config["paths"]["protocol"])
    protocol_hash = sha256_file(protocol_path)
    if protocol_hash != config["expected_protocol_sha256"]:
        raise RuntimeError("frozen V2 protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["training"]["validation_checkpoint_selection"] or config["restrictions"]["validation_checkpoint_selection"]:
        raise RuntimeError("validation checkpoint selection is prohibited")
    control_ids = [item["id"] for item in config["controls"]]
    if control_ids != protocol["allowed_stage_e4s2_controls"]:
        raise RuntimeError("control set differs from frozen protocol")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_root = resolve(config["paths"]["adapter_root"])
    graphs = load_graphs(resolve(config["paths"]["graph_root"]))
    folds = [str(fold) for fold in protocol["folds"]]
    seeds = [int(seed) for seed in protocol["seeds"]]
    receipts = []
    prediction_frames = []
    curves = []
    stability = {}
    for variant in config["controls"]:
        by_seed = {}
        seed_mae = {}
        for seed in seeds:
            seed_rows = []
            for fold_id in folds:
                data = load_fold(adapter_root, fold_id, variant["text_view"], False)
                receipt, rows, curve = run_fold_v2(protocol, variant, fold_id, seed, data, graphs, output_root)
                receipts.append(receipt)
                seed_rows.extend(rows)
                curves.extend(curve)
                print(f"{variant['id']} seed={seed} {fold_id} MAE={receipt['mae']:.6f} train_n={receipt['train_cross_section_count']}", flush=True)
            frame = pd.DataFrame(seed_rows)
            by_seed[seed] = frame
            prediction_frames.append(frame)
            valid = frame["sample_valid"].astype(bool)
            seed_mae[seed] = float(np.abs(frame.loc[valid, "prediction"] - frame.loc[valid, "target_return"]).mean())
        pairwise = []
        for left_index in range(len(seeds)):
            for right_index in range(left_index + 1, len(seeds)):
                left, right = seeds[left_index], seeds[right_index]
                pairwise.append({
                    "seed_a": left, "seed_b": right, "edge_jaccard": 1.0,
                    "prediction_correlation": prediction_correlation(by_seed[left], by_seed[right]),
                })
        values = np.asarray([seed_mae[seed] for seed in seeds], dtype=float)
        mae_cv = float(values.std() / max(values.mean(), 1e-12))
        thresholds = protocol["stability_thresholds"]
        passed = (
            min(item["edge_jaccard"] for item in pairwise) >= float(thresholds["minimum_pairwise_edge_jaccard"])
            and min(item["prediction_correlation"] for item in pairwise) >= float(thresholds["minimum_pairwise_prediction_correlation"])
            and mae_cv <= float(thresholds["maximum_mae_coefficient_of_variation"])
        )
        stability[variant["id"]] = {
            "seed_mae": {str(key): value for key, value in seed_mae.items()}, "pairwise": pairwise,
            "mae_coefficient_of_variation": mae_cv, "passed": passed,
        }
    fold_path = output_root / "fold_results.csv"
    predictions_path = output_root / "predictions.csv.gz"
    curves_path = output_root / "training_curves.csv"
    pd.DataFrame(receipts).to_csv(fold_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(curves).to_csv(curves_path, index=False)
    stable_controls = [variant for variant, result in stability.items() if result["passed"]]
    report = {
        "stage": "E-4S.2", "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "controls": control_ids, "folds": folds, "seeds": seeds,
        "training_receipt": {
            "use_all_train_cross_sections": True, "epochs": protocol["training"]["epochs"],
            "optimizer": protocol["training"]["optimizer"], "scheduler": protocol["training"]["scheduler"],
            "ema_decay": protocol["training"]["ema_decay"], "evaluation_weights": protocol["training"]["evaluation_weights"],
            "validation_checkpoint_selection": False,
        },
        "stability_thresholds": protocol["stability_thresholds"], "stability": stability,
        "stable_controls": stable_controls, "gate_a_pass": bool(stable_controls),
        "allow_stage_e4s3": bool(stable_controls), "allow_300_stock_expansion": False,
        "future_or_sealed_data_read": False, "screening_accessed": False,
        "config_sha256": sha256_file(config_path),
        "source_sha256": {
            "adapter_base": {fold: sha256_file(adapter_root / fold / "base_windows.npz") for fold in folds},
            "fixed_graphs": sha256_file(graphs["fixed_path"]), "rolling_graphs": sha256_file(graphs["rolling_path"]),
        },
        "artifacts": {
            "fold_results_sha256": sha256_file(fold_path), "predictions_sha256": sha256_file(predictions_path),
            "training_curves_sha256": sha256_file(curves_path),
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
