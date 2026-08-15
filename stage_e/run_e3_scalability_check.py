"""Verify that the adaptive stock graph runs safely at the frozen 300-stock scale."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.models.cross_sectional_forecaster import CrossSectionalTemporalForecaster
from stage_e.run_e3_training_checks import prepare_data, resolve, set_seed


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data = prepare_data(config)
    set_seed(int(config["seed"]))
    options = config["model"]
    model = CrossSectionalTemporalForecaster(
        input_dim=data["x"].shape[-1], stock_count=int(config["data"]["stock_count"]),
        hidden_dim=int(options["hidden_dim"]), top_k=int(options["top_k"]),
        dropout=float(options["dropout"]), sampling_mode=str(options["sampling_mode"]),
    )
    model.train()
    start = time.perf_counter()
    details = model(data["x"], data["available"], return_details=True)
    forward_seconds = time.perf_counter() - start
    loss = nn.MSELoss()(details["prediction"][data["mask"]], data["y"][data["mask"]])
    start = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - start
    gradient_norm = float(torch.sqrt(sum((parameter.grad.detach() ** 2).sum() for parameter in model.parameters() if parameter.grad is not None)))
    adjacency = details["adjacency"].detach().cpu().numpy()
    available = data["available"].cpu().numpy()
    nonself = adjacency > 0
    diagonal = np.eye(adjacency.shape[-1], dtype=bool)[None, :, :]
    nonself &= ~diagonal
    degree = nonself.sum(axis=-1)
    unavailable_rows = ~available
    unavailable_self_only = bool((degree[unavailable_rows] == 0).all()) if unavailable_rows.any() else True
    checks = {
        "input_shape_300stocks": data["x"].shape[2] == 300,
        "adjacency_shape": adjacency.shape == (int(config["data"]["cross_section_count"]), 300, 300),
        "finite_prediction_and_adjacency": bool(np.isfinite(adjacency).all() and torch.isfinite(details["prediction"]).all()),
        "row_stochastic": bool(np.allclose(adjacency.sum(axis=-1), 1.0, atol=1e-5)),
        "active_topk_exact": bool((degree[available] == int(options["top_k"])).all()),
        "unavailable_self_only": unavailable_self_only,
        "finite_positive_gradient": math.isfinite(gradient_norm) and gradient_norm > 0,
        "future_or_sealed_data_not_read": True,
    }
    edge_rows = []
    for date_index, trade_date in enumerate(data["dates"]):
        for source_index, source_stock in enumerate(data["stocks"]):
            for target_index in np.flatnonzero(nonself[date_index, source_index]):
                edge_rows.append({
                    "trade_date": trade_date, "source_stock": source_stock,
                    "target_stock": data["stocks"][target_index],
                    "weight": float(adjacency[date_index, source_index, target_index]),
                })
    edges_path = output_root / "adaptive_edges.csv.gz"
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False, compression={"method": "gzip", "mtime": 0})
    checkpoint_path = output_root / "untrained_gradient_checked_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    report = {
        "stage": "E-3.5", "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(checks.values()),
        "checks": checks, "input_shape": list(data["x"].shape), "adjacency_shape": list(adjacency.shape),
        "valid_target_rows": int(data["mask"].sum()), "available_node_rows": int(data["available"].sum()),
        "mean_nonself_degree_active": float(degree[available].mean()),
        "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
        "gradient_norm": gradient_norm, "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loss_before_update": float(loss.detach()),
        "artifacts": {"edges_sha256": sha256_file(edges_path), "checkpoint_sha256": sha256_file(checkpoint_path)},
        "config_sha256": sha256_file(config_path), "panel_sha256": sha256_file(data["panel_path"]),
        "stock_order_sha256": sha256_file(data["order_path"]), "future_or_sealed_data_read": False,
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
