"""Run 100-stock E-4 graph-frequency tiny overfit and three-seed initial checks."""

from __future__ import annotations

import argparse
import json
import sys
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
from stage_e.run_e3_training_checks import pairwise_correlation, pairwise_jaccard, resolve, set_seed


def load_subset(config: dict[str, Any]) -> dict[str, Any]:
    root = resolve(config["paths"]["adapter_root"]) / config["fold_id"]
    base_path = root / "base_windows.npz"
    base = np.load(base_path)
    train_indices = np.flatnonzero(base["split"].astype(str) == "train")[: int(config["train_cross_sections"])]
    if len(train_indices) != int(config["train_cross_sections"]):
        raise ValueError("insufficient TRAIN cross-sections for E-4 checks")
    return {
        "x": torch.tensor(base["values"][train_indices], dtype=torch.float32),
        "y": torch.tensor(base["target_scaled"][train_indices], dtype=torch.float32),
        "raw_y": base["target_raw"][train_indices],
        "mask": torch.tensor(base["sample_mask"][train_indices], dtype=torch.bool),
        "available": torch.tensor(base["node_available"][train_indices], dtype=torch.bool),
        "dates": base["trade_date"][train_indices].astype(str).tolist(),
        "stocks": base["stock_code"].astype(str).tolist(),
        "target_mean": float(base["target_mean_train"][0]), "target_std": float(base["target_std_train"][0]),
        "base_path": base_path,
    }


def train_once(data: dict[str, Any], config: dict[str, Any], seed: int, epochs: int, lr: float, weight_decay: float) -> dict[str, Any]:
    set_seed(seed)
    options = config["model"]
    model = GraphFrequencyFusionModel(
        input_dim=data["x"].shape[-1], stock_count=data["x"].shape[2], hidden_dim=int(options["hidden_dim"]),
        top_k=int(options["top_k"]), dropout=float(options["dropout"]), graph_mode=options["graph_mode"],
        branch_mode=options["branch_mode"], fusion_mode=options["fusion_mode"], text_fusion=options["text_fusion"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    x, y, mask, available = data["x"], data["y"], data["mask"], data["available"]
    model.eval()
    with torch.no_grad():
        initial = model(x, node_available=available)
        initial_loss = float(loss_function(initial[mask], y[mask]))
    curve = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, node_available=available)
        loss = loss_function(prediction[mask], y[mask])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite E-4 loss for seed {seed}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            curve.append({"seed": seed, "epoch": epoch, "loss": float(loss.detach())})
    model.eval()
    with torch.no_grad():
        details = model(x, node_available=available, return_details=True)
    scaled_prediction = details["prediction"].cpu().numpy()
    raw_prediction = scaled_prediction * data["target_std"] + data["target_mean"]
    raw_mask = mask.cpu().numpy()
    final_loss = float(loss_function(details["prediction"][mask], y[mask]))
    raw_mae = float(np.mean(np.abs(raw_prediction[raw_mask] - data["raw_y"][raw_mask])))
    adjacency = details["adjacency"].cpu().numpy()
    edge_mask = adjacency > 0
    diagonal = np.eye(adjacency.shape[-1], dtype=bool)[None, :, :]
    edge_mask &= ~diagonal
    degree = edge_mask.sum(axis=-1)
    gate = details["gate"]
    return {
        "seed": seed, "model": model, "curve": curve, "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_reduction": 1.0 - final_loss / max(initial_loss, 1e-12), "raw_mae": raw_mae,
        "prediction": raw_prediction, "adjacency": adjacency, "edge_mask": edge_mask,
        "finite": bool(np.isfinite(raw_prediction).all() and np.isfinite(adjacency).all()),
        "row_stochastic": bool(np.allclose(adjacency.sum(axis=-1), 1.0, atol=1e-5)),
        "active_topk_exact": bool((degree[data["available"].cpu().numpy()] == int(options["top_k"])).all()),
        "mean_degree": float(degree.mean()), "gate_mean": None if gate is None else float(gate.mean()),
    }


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data = load_subset(config)
    overfit_options = config["overfit"]
    overfit = train_once(data, config, int(overfit_options["seed"]), int(overfit_options["epochs"]), float(overfit_options["learning_rate"]), float(overfit_options["weight_decay"]))
    overfit_pass = (
        overfit["loss_reduction"] >= float(overfit_options["required_loss_reduction"])
        and overfit["raw_mae"] <= float(overfit_options["maximum_raw_mae"])
        and overfit["finite"] and overfit["row_stochastic"] and overfit["active_topk_exact"]
    )
    stability_options = config["stability"]
    seeds = [
        train_once(data, config, int(seed), int(stability_options["epochs"]), float(stability_options["learning_rate"]), float(stability_options["weight_decay"]))
        for seed in stability_options["seeds"]
    ]
    mask = data["mask"].cpu().numpy()
    pairwise = []
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            pairwise.append({
                "seed_a": seeds[left]["seed"], "seed_b": seeds[right]["seed"],
                "edge_jaccard": pairwise_jaccard(seeds[left]["edge_mask"], seeds[right]["edge_mask"]),
                "prediction_correlation": pairwise_correlation(seeds[left]["prediction"], seeds[right]["prediction"], mask),
                "adjacency_mean_absolute_difference": float(np.abs(seeds[left]["adjacency"] - seeds[right]["adjacency"]).mean()),
            })
    maes = np.asarray([item["raw_mae"] for item in seeds])
    mae_cv = float(maes.std() / max(maes.mean(), 1e-12))
    stability_pass = (
        all(item["finite"] and item["row_stochastic"] and item["active_topk_exact"] for item in seeds)
        and min(item["edge_jaccard"] for item in pairwise) >= float(stability_options["minimum_pairwise_edge_jaccard"])
        and min(item["prediction_correlation"] for item in pairwise) >= float(stability_options["minimum_pairwise_prediction_correlation"])
        and mae_cv <= float(stability_options["maximum_mae_coefficient_of_variation"])
    )
    curve_path = output_root / "training_curves.csv"
    pd.DataFrame(overfit["curve"] + [row for result in seeds for row in result["curve"]]).to_csv(curve_path, index=False)
    checkpoints = []
    for result in seeds:
        path = output_root / f"model_seed_{result['seed']}.pt"
        torch.save(result["model"].state_dict(), path)
        checkpoints.append({"seed": result["seed"], "path": path.name, "sha256": sha256_file(path)})
    report = {
        "stage": "E-4.3", "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_shape": list(data["x"].shape), "dates": data["dates"], "stock_count": len(data["stocks"]),
        "overfit": {key: overfit[key] for key in ("seed", "initial_loss", "final_loss", "loss_reduction", "raw_mae", "finite", "row_stochastic", "active_topk_exact", "mean_degree", "gate_mean")},
        "overfit_pass": overfit_pass,
        "stability_seeds": [{key: item[key] for key in ("seed", "initial_loss", "final_loss", "loss_reduction", "raw_mae", "finite", "row_stochastic", "active_topk_exact", "mean_degree", "gate_mean")} for item in seeds],
        "pairwise": pairwise, "mae_coefficient_of_variation": mae_cv,
        "stability_pass": stability_pass, "passed": overfit_pass and stability_pass,
        "artifacts": {"training_curves_sha256": sha256_file(curve_path), "checkpoints": checkpoints},
        "config_sha256": sha256_file(config_path), "adapter_base_sha256": sha256_file(data["base_path"]),
        "future_or_sealed_data_read": False,
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
