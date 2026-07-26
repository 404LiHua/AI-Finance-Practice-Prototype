"""Run the preregistered bounded E-4 graph/frequency/text ablations."""

from __future__ import annotations

import argparse
import copy
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


def load_graphs(graph_root: Path) -> dict[str, Any]:
    fixed_path = graph_root / "fixed_graphs.npz"
    rolling_path = graph_root / "rolling_correlation_graphs.npz"
    fixed = np.load(fixed_path)
    rolling = np.load(rolling_path)
    return {
        "identity": fixed["identity"].astype(np.float32), "industry": fixed["industry"].astype(np.float32),
        "rolling": rolling["adjacency"].astype(np.float32),
        "rolling_dates": {str(date): index for index, date in enumerate(rolling["trade_dates"].astype(str))},
        "fixed_path": fixed_path, "rolling_path": rolling_path,
    }


def load_fold(adapter_root: Path, fold_id: str, text_view: str, include_markers: bool) -> dict[str, Any]:
    fold_root = adapter_root / fold_id
    base_path = fold_root / "base_windows.npz"
    text_path = fold_root / f"text_{text_view}.npz"
    base = np.load(base_path)
    text = np.load(text_path)
    features = text["features"].astype(np.float32)
    if text_view != "no_text" and include_markers:
        markers = np.stack([
            text["text_available"].astype(np.float32), np.log1p(text["text_count"].astype(np.float32)),
        ], axis=-1)
        features = np.concatenate([features, markers], axis=-1)
    return {
        "x": base["values"].astype(np.float32), "target_scaled": base["target_scaled"].astype(np.float32),
        "target_raw": base["target_raw"].astype(np.float32), "mask": base["sample_mask"].astype(bool),
        "available": base["node_available"].astype(bool), "split": base["split"].astype(str),
        "dates": base["trade_date"].astype(str), "stocks": base["stock_code"].astype(str),
        "target_mean": float(base["target_mean_train"][0]), "target_std": float(base["target_std_train"][0]),
        "text": features, "text_available": text["text_available"].astype(bool),
        "base_path": base_path, "text_path": text_path,
    }


def provided_adjacency(graph_kind: str, indices: np.ndarray, dates: np.ndarray, graphs: dict[str, Any]) -> torch.Tensor | None:
    if graph_kind == "industry":
        return torch.tensor(np.broadcast_to(graphs["industry"], (len(indices), *graphs["industry"].shape)).copy(), dtype=torch.float32)
    if graph_kind == "rolling_correlation":
        values = np.stack([graphs["rolling"][graphs["rolling_dates"][str(dates[index])]] for index in indices])
        return torch.tensor(values, dtype=torch.float32)
    return None


def graph_mode(kind: str) -> str:
    if kind == "no_graph":
        return "no_graph"
    if kind in {"industry", "rolling_correlation"}:
        return "provided"
    return kind


def raw_metrics(prediction_scaled: torch.Tensor, data: dict[str, Any], indices: np.ndarray) -> dict[str, float]:
    prediction = prediction_scaled.detach().cpu().numpy() * data["target_std"] + data["target_mean"]
    target = data["target_raw"][indices]
    mask = data["mask"][indices]
    errors = prediction[mask] - target[mask]
    return {
        "mae": float(np.mean(np.abs(errors))), "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "direction_accuracy": float(np.mean((prediction[mask] >= 0) == (target[mask] >= 0))),
        "sample_count": int(mask.sum()),
    }


def run_fold(
    config: dict[str, Any], variant: dict[str, str], fold_id: str, data: dict[str, Any], graphs: dict[str, Any], output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(int(config["seed"]))
    train_options = config["training"]
    all_train = np.flatnonzero(data["split"] == "train")
    train_indices = all_train[-int(train_options["train_cross_section_cap"]):]
    validation_indices = np.flatnonzero(data["split"] == "validation")
    text_dim = data["text"].shape[-1]
    model = GraphFrequencyFusionModel(
        input_dim=data["x"].shape[-1], stock_count=data["x"].shape[2], hidden_dim=int(train_options["hidden_dim"]),
        top_k=int(train_options["top_k"]), dropout=float(train_options["dropout"]), graph_mode=graph_mode(variant["graph"]),
        branch_mode=variant["branch"], fusion_mode=variant["fusion"], text_dim=text_dim,
        text_fusion=variant["text_fusion"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_options["learning_rate"]), weight_decay=float(train_options["weight_decay"]))
    loss_function = nn.SmoothL1Loss()
    generator = np.random.default_rng(int(config["seed"]))
    best_state = copy.deepcopy(model.state_dict())
    best_mae = math.inf
    best_epoch = 0
    patience = 0
    started = time.perf_counter()
    for epoch in range(1, int(train_options["epochs"]) + 1):
        shuffled = generator.permutation(train_indices)
        model.train()
        for start in range(0, len(shuffled), int(train_options["batch_size"])):
            batch = shuffled[start : start + int(train_options["batch_size"])]
            x = torch.tensor(data["x"][batch], dtype=torch.float32)
            y = torch.tensor(data["target_scaled"][batch], dtype=torch.float32)
            mask = torch.tensor(data["mask"][batch], dtype=torch.bool)
            available = torch.tensor(data["available"][batch], dtype=torch.bool)
            text = None if variant["text_view"] == "no_text" else torch.tensor(data["text"][batch], dtype=torch.float32)
            adjacency = provided_adjacency(variant["graph"], batch, data["dates"], graphs)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, node_available=available, adjacency=adjacency, text_features=text)
            loss = loss_function(prediction[mask], y[mask])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {variant['id']} {fold_id}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_options["gradient_clip"]))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(data["x"][validation_indices], dtype=torch.float32)
            val_available = torch.tensor(data["available"][validation_indices], dtype=torch.bool)
            val_text = None if variant["text_view"] == "no_text" else torch.tensor(data["text"][validation_indices], dtype=torch.float32)
            val_adjacency = provided_adjacency(variant["graph"], validation_indices, data["dates"], graphs)
            val_prediction = model(val_x, node_available=val_available, adjacency=val_adjacency, text_features=val_text)
            metrics = raw_metrics(val_prediction, data, validation_indices)
        if metrics["mae"] < best_mae - 1e-8:
            best_mae = metrics["mae"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(train_options["patience"]):
                break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(data["x"][validation_indices], dtype=torch.float32)
        val_available = torch.tensor(data["available"][validation_indices], dtype=torch.bool)
        val_text = None if variant["text_view"] == "no_text" else torch.tensor(data["text"][validation_indices], dtype=torch.float32)
        val_adjacency = provided_adjacency(variant["graph"], validation_indices, data["dates"], graphs)
        details = model(val_x, node_available=val_available, adjacency=val_adjacency, text_features=val_text, return_details=True)
    metrics = raw_metrics(details["prediction"], data, validation_indices)
    adjacency_array = details["adjacency"].cpu().numpy()
    gate = details["gate"]
    checkpoint = output_root / "checkpoints" / f"{variant['id']}__{fold_id}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    prediction_raw = details["prediction"].cpu().numpy() * data["target_std"] + data["target_mean"]
    prediction_rows = []
    for local, index in enumerate(validation_indices):
        for stock_index, stock in enumerate(data["stocks"]):
            prediction_rows.append({
                "variant": variant["id"], "fold_id": fold_id, "trade_date": data["dates"][index],
                "stock_code": stock, "target_return": float(data["target_raw"][index, stock_index]),
                "prediction": float(prediction_raw[local, stock_index]), "sample_valid": bool(data["mask"][index, stock_index]),
                "text_available": bool(data["text_available"][index, stock_index]),
            })
    receipt = {
        "variant": variant["id"], "fold_id": fold_id, **metrics, "best_epoch": best_epoch,
        "training_seconds": training_seconds, "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "adjacency_finite": bool(np.isfinite(adjacency_array).all()),
        "adjacency_row_stochastic": bool(np.allclose(adjacency_array.sum(axis=-1), 1.0, atol=1e-5)),
        "mean_nonself_degree": float(((adjacency_array > 0).sum(axis=-1) - 1).mean()),
        "gate_mean": None if gate is None else float(gate.mean()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    return receipt, prediction_rows


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter_root = resolve(config["paths"]["adapter_root"])
    graph_root = resolve(config["paths"]["graph_root"])
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    graphs = load_graphs(graph_root)
    fold_ids = ["E_RO_01", "E_RO_02", "E_RO_03"]
    receipts = []
    predictions = []
    source_hashes = {}
    for variant in config["variants"]:
        for fold_id in fold_ids:
            data = load_fold(adapter_root, fold_id, variant["text_view"], bool(config["text"]["include_availability_and_log_count"]))
            source_hashes[f"{fold_id}/base"] = sha256_file(data["base_path"])
            source_hashes[f"{fold_id}/{variant['text_view']}"] = sha256_file(data["text_path"])
            receipt, rows = run_fold(config, variant, fold_id, data, graphs, output_root)
            receipts.append(receipt)
            predictions.extend(rows)
            print(f"{variant['id']} {fold_id} MAE={receipt['mae']:.6f} epoch={receipt['best_epoch']}", flush=True)
    predictions_path = output_root / "validation_predictions.csv.gz"
    pd.DataFrame(predictions).to_csv(predictions_path, index=False, compression={"method": "gzip", "mtime": 0})
    receipts_frame = pd.DataFrame(receipts)
    results_path = output_root / "fold_results.csv"
    receipts_frame.to_csv(results_path, index=False)
    summary = receipts_frame.groupby("variant", as_index=False).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_mse=("mse", "mean"),
        mean_direction_accuracy=("direction_accuracy", "mean"), worst_fold_mae=("mae", "max"),
        total_training_seconds=("training_seconds", "sum"), parameter_count=("parameter_count", "max"),
        adjacency_valid=("adjacency_finite", "all"), row_stochastic=("adjacency_row_stochastic", "all"),
    ).sort_values("mean_mae", kind="stable")
    summary_path = output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    temporal_mae = float(summary.loc[summary["variant"].eq("temporal_no_graph_no_text"), "mean_mae"].iloc[0])
    dual_price_mae = float(summary.loc[summary["variant"].eq("dual_learned_fixed_no_text"), "mean_mae"].iloc[0])
    comparisons = {
        "price_variants_vs_temporal_mae_delta": {
            row.variant: float(row.mean_mae - temporal_mae) for row in summary.itertuples() if "no_text" in row.variant
        },
        "text_variants_vs_dual_price_mae_delta": {
            row.variant: float(row.mean_mae - dual_price_mae) for row in summary.itertuples() if "tfidf" in row.variant or "bge" in row.variant
        },
    }
    report = {
        "stage": "E-4.4", "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "seed": config["seed"],
        "variant_count": len(config["variants"]), "fold_run_count": len(receipts),
        "selection_exposure": "TRAIN/VALIDATION development ablation; no candidate promotion and no future data read",
        "comparisons": comparisons,
        "artifacts": {"fold_results_sha256": sha256_file(results_path), "summary_sha256": sha256_file(summary_path), "predictions_sha256": sha256_file(predictions_path)},
        "source_hashes": source_hashes, "fixed_graphs_sha256": sha256_file(graphs["fixed_path"]),
        "rolling_graphs_sha256": sha256_file(graphs["rolling_path"]), "config_sha256": sha256_file(config_path),
        "future_or_sealed_data_read": False,
    }
    report["batch_sha256"] = stable_json_sha256(report)
    (output_root / "metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
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
