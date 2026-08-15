"""Run E-3 tiny overfit and three-seed structural stability checks."""

from __future__ import annotations

import argparse
import json
import math
import random
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

from stage_e.custody import StageEDataCustodyGuard
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.models.cross_sectional_forecaster import CrossSectionalTemporalForecaster

DEFAULT_CUSTODY = REPO_ROOT / "stage_e/configs/data_custody_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    guard = StageEDataCustodyGuard.from_config(DEFAULT_CUSTODY, REPO_ROOT)
    panel_path = guard.assert_path_allowed(resolve(config["paths"]["panel"]), purpose="E-3 training panel")
    order_path = guard.assert_path_allowed(resolve(config["paths"]["stock_order"]), purpose="E-3 stock order")
    options = config["data"]
    features = list(options["feature_columns"])
    usecols = ["trade_date", "stock_code", "is_market_open_week", "model_eligible_pit", "sample_eligible_v2", options["target_column"], *features]
    panel = pd.read_csv(panel_path, usecols=usecols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    guard.assert_development_frame(panel, date_columns=("trade_date",))
    order = pd.read_csv(order_path).sort_values("selection_rank", kind="stable")
    stocks = order["stock_code"].astype(str).head(int(options["stock_count"])).tolist()
    panel = panel[panel["stock_code"].astype(str).isin(stocks)].copy()
    dates = sorted(panel.loc[panel["is_market_open_week"].astype(bool), "trade_date"].dropna().unique())
    index = pd.MultiIndex.from_product([dates, stocks], names=["trade_date", "stock_code"])
    dense = panel.set_index(["trade_date", "stock_code"]).reindex(index)
    feature_arrays = []
    for column in features:
        values = pd.to_numeric(dense[column], errors="coerce").to_numpy(dtype=np.float64).reshape(len(dates), len(stocks))
        if column == "model_close":
            values = np.log(np.clip(values, 1e-8, None))
        elif column == "model_volume_hands":
            values = np.log1p(np.clip(values, 0.0, None))
        feature_arrays.append(values)
    feature_cube = np.stack(feature_arrays, axis=-1)
    target = pd.to_numeric(dense[options["target_column"]], errors="coerce").to_numpy(dtype=np.float64).reshape(len(dates), len(stocks))
    eligible = dense["sample_eligible_v2"].fillna(False).astype(bool).to_numpy().reshape(len(dates), len(stocks))
    available = dense["model_eligible_pit"].fillna(False).astype(bool).to_numpy().reshape(len(dates), len(stocks))
    lookback = int(options["lookback_weeks"])
    candidates = []
    for current in range(lookback - 1, len(dates)):
        target_mask = eligible[current] & np.isfinite(target[current])
        if target_mask.mean() >= float(options["minimum_target_coverage"]):
            candidates.append(current)
    selected = candidates[: int(options["cross_section_count"])]
    if len(selected) < int(options["cross_section_count"]):
        raise ValueError("insufficient eligible cross-sections for E-3 training checks")
    x = np.stack([feature_cube[current - lookback + 1 : current + 1] for current in selected])
    y = np.stack([target[current] for current in selected])
    mask = np.stack([eligible[current] & np.isfinite(target[current]) for current in selected])
    node_available = np.stack([available[current] for current in selected])
    finite = np.isfinite(x)
    means = np.asarray([x[..., index][finite[..., index]].mean() for index in range(x.shape[-1])])
    stds = np.asarray([x[..., index][finite[..., index]].std() for index in range(x.shape[-1])])
    stds = np.where(stds < 1e-8, 1.0, stds)
    x = (x - means) / stds
    x[~np.isfinite(x)] = 0.0
    target_values = y[mask]
    target_mean = float(target_values.mean())
    target_std = float(target_values.std())
    if target_std < 1e-8:
        raise ValueError("target standard deviation is too small")
    y_scaled = (y - target_mean) / target_std
    y_scaled[~mask] = 0.0
    return {
        "x": torch.tensor(x, dtype=torch.float32),
        "y": torch.tensor(y_scaled, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "available": torch.tensor(node_available, dtype=torch.bool),
        "raw_y": y, "dates": [pd.Timestamp(dates[index]).strftime("%Y-%m-%d") for index in selected],
        "stocks": stocks, "target_mean": target_mean, "target_std": target_std,
        "feature_means": means.tolist(), "feature_stds": stds.tolist(),
        "panel_path": panel_path, "order_path": order_path,
    }


def train_once(data: dict[str, Any], config: dict[str, Any], seed: int, epochs: int, learning_rate: float, weight_decay: float) -> dict[str, Any]:
    set_seed(seed)
    model_options = config["model"]
    model = CrossSectionalTemporalForecaster(
        input_dim=data["x"].shape[-1], stock_count=data["x"].shape[2],
        hidden_dim=int(model_options["hidden_dim"]), top_k=int(model_options["top_k"]),
        dropout=float(model_options["dropout"]), sampling_mode=str(model_options["sampling_mode"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    x, y, mask, available = data["x"], data["y"], data["mask"], data["available"]
    model.eval()
    with torch.no_grad():
        initial = model(x, available)
        initial_loss = float(loss_function(initial[mask], y[mask]))
    curve = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, available)
        loss = loss_function(prediction[mask], y[mask])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss for seed {seed}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            curve.append({"seed": seed, "epoch": epoch, "loss": float(loss.detach())})
    model.eval()
    with torch.no_grad():
        details = model(x, available, return_details=True)
    scaled_prediction = details["prediction"].cpu().numpy()
    raw_prediction = scaled_prediction * data["target_std"] + data["target_mean"]
    raw_mask = mask.cpu().numpy()
    final_loss = float(loss_function(details["prediction"][mask], y[mask]))
    raw_mae = float(np.mean(np.abs(raw_prediction[raw_mask] - data["raw_y"][raw_mask])))
    adjacency = details["adjacency"].cpu().numpy()
    degrees = (adjacency > 0).sum(axis=-1) - 1
    finite = bool(np.isfinite(adjacency).all() and np.isfinite(raw_prediction).all())
    row_stochastic = bool(np.allclose(adjacency.sum(axis=-1), 1.0, atol=1e-5))
    isolated = int((degrees < 1).sum())
    edge_mask = adjacency > 0
    diagonal = np.eye(adjacency.shape[-1], dtype=bool)[None, :, :]
    edge_mask = edge_mask & ~diagonal
    return {
        "seed": seed, "model": model, "curve": curve, "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_reduction": 1.0 - final_loss / max(initial_loss, 1e-12), "raw_mae": raw_mae,
        "prediction": raw_prediction, "adjacency": adjacency, "edge_mask": edge_mask,
        "finite": finite, "row_stochastic": row_stochastic, "isolated_node_rows": isolated,
        "mean_nonself_degree": float(degrees.mean()), "degree_std": float(degrees.std()),
    }


def pairwise_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 1.0


def pairwise_correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    a, b = first[mask], second[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data = prepare_data(config)
    overfit_options = config["overfit"]
    overfit = train_once(
        data, config, int(overfit_options["seed"]), int(overfit_options["epochs"]),
        float(overfit_options["learning_rate"]), float(overfit_options["weight_decay"]),
    )
    overfit_pass = (
        overfit["loss_reduction"] >= float(overfit_options["required_loss_reduction"])
        and overfit["raw_mae"] <= float(overfit_options["maximum_raw_mae"])
        and overfit["finite"] and overfit["row_stochastic"] and overfit["isolated_node_rows"] == 0
    )
    stability_options = config["stability"]
    seed_results = [
        train_once(data, config, int(seed), int(stability_options["epochs"]), float(stability_options["learning_rate"]), float(stability_options["weight_decay"]))
        for seed in stability_options["seeds"]
    ]
    pairwise = []
    mask = data["mask"].cpu().numpy()
    for left in range(len(seed_results)):
        for right in range(left + 1, len(seed_results)):
            pairwise.append({
                "seed_a": seed_results[left]["seed"], "seed_b": seed_results[right]["seed"],
                "edge_jaccard": pairwise_jaccard(seed_results[left]["edge_mask"], seed_results[right]["edge_mask"]),
                "prediction_correlation": pairwise_correlation(seed_results[left]["prediction"], seed_results[right]["prediction"], mask),
                "mean_adjacency_absolute_difference": float(np.abs(seed_results[left]["adjacency"] - seed_results[right]["adjacency"]).mean()),
            })
    maes = np.asarray([item["raw_mae"] for item in seed_results])
    mae_cv = float(maes.std() / max(maes.mean(), 1e-12))
    stability_pass = (
        all(item["finite"] and item["row_stochastic"] and item["isolated_node_rows"] == 0 for item in seed_results)
        and min(item["edge_jaccard"] for item in pairwise) >= float(stability_options["minimum_pairwise_edge_jaccard"])
        and min(item["prediction_correlation"] for item in pairwise) >= float(stability_options["minimum_pairwise_prediction_correlation"])
        and mae_cv <= float(stability_options["maximum_mae_coefficient_of_variation"])
    )
    curves = pd.DataFrame(overfit["curve"] + [row for result in seed_results for row in result["curve"]])
    curves_path = output_root / "training_curves.csv"
    curves.to_csv(curves_path, index=False)
    prediction_rows = []
    edge_rows = []
    for result in seed_results:
        for date_index, trade_date in enumerate(data["dates"]):
            for stock_index, stock_code in enumerate(data["stocks"]):
                prediction_rows.append({
                    "seed": result["seed"], "trade_date": trade_date, "stock_code": stock_code,
                    "target_return": float(data["raw_y"][date_index, stock_index]),
                    "prediction": float(result["prediction"][date_index, stock_index]),
                    "target_valid": bool(mask[date_index, stock_index]),
                })
                targets = np.flatnonzero(result["edge_mask"][date_index, stock_index])
                for target_index in targets:
                    edge_rows.append({
                        "seed": result["seed"], "trade_date": trade_date, "source_stock": stock_code,
                        "target_stock": data["stocks"][target_index],
                        "weight": float(result["adjacency"][date_index, stock_index, target_index]),
                    })
    predictions_path = output_root / "seed_predictions.csv.gz"
    edges_path = output_root / "seed_edges.csv.gz"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False, compression={"method": "gzip", "mtime": 0})
    checkpoints = []
    for result in seed_results:
        path = output_root / f"model_seed_{result['seed']}.pt"
        torch.save(result["model"].state_dict(), path)
        checkpoints.append({"seed": result["seed"], "path": path.name, "sha256": sha256_file(path)})
    report = {
        "stage": "E-3.4", "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_date_ceiling": config["development_date_ceiling"],
        "data": {"stocks": data["stocks"], "dates": data["dates"], "lookback_weeks": config["data"]["lookback_weeks"], "target_mean": data["target_mean"], "target_std": data["target_std"]},
        "overfit": {key: overfit[key] for key in ("seed", "initial_loss", "final_loss", "loss_reduction", "raw_mae", "finite", "row_stochastic", "isolated_node_rows", "mean_nonself_degree", "degree_std")},
        "overfit_pass": overfit_pass,
        "stability_seeds": [{key: item[key] for key in ("seed", "initial_loss", "final_loss", "loss_reduction", "raw_mae", "finite", "row_stochastic", "isolated_node_rows", "mean_nonself_degree", "degree_std")} for item in seed_results],
        "pairwise": pairwise, "mae_coefficient_of_variation": mae_cv,
        "stability_pass": stability_pass, "passed": overfit_pass and stability_pass,
        "artifacts": {
            "training_curves_sha256": sha256_file(curves_path), "predictions_sha256": sha256_file(predictions_path),
            "edges_sha256": sha256_file(edges_path), "checkpoints": checkpoints,
        },
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
