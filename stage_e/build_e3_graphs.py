"""Build auditable identity, industry and rolling-correlation stock graphs for E-3."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.custody import StageEDataCustodyGuard
from stage_e.hashing import sha256_file, stable_json_sha256

DEFAULT_CUSTODY = REPO_ROOT / "stage_e/configs/data_custody_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def row_normalize(adjacency: np.ndarray) -> np.ndarray:
    row_sum = adjacency.sum(axis=-1, keepdims=True)
    return adjacency / np.clip(row_sum, 1e-12, None)


def build_industry_graph(industries: list[str], options: dict[str, Any]) -> np.ndarray:
    count = len(industries)
    unknown = {str(value).strip() for value in options.get("unknown_groups", [])}
    graph = np.zeros((count, count), dtype=np.float32)
    for row, industry in enumerate(industries):
        label = str(industry).strip()
        if options.get("unknown_self_only", True) and label in unknown:
            graph[row, row] = float(options.get("self_weight", 1.0))
            continue
        for column, other in enumerate(industries):
            if label == str(other).strip():
                graph[row, column] = 1.0
        graph[row, row] = max(graph[row, row], float(options.get("self_weight", 1.0)))
    return row_normalize(graph).astype(np.float32)


def correlation_topk(correlation: np.ndarray, top_k: int, minimum: float, self_weight: float) -> np.ndarray:
    count = correlation.shape[0]
    graph = np.zeros((count, count), dtype=np.float32)
    indices = np.arange(count)
    for row in range(count):
        scores = np.abs(correlation[row]).astype(float)
        scores[~np.isfinite(scores)] = -np.inf
        scores[row] = -np.inf
        order = np.lexsort((indices, -scores))
        selected = [index for index in order if scores[index] >= minimum][:top_k]
        graph[row, row] = self_weight
        if selected:
            graph[row, selected] = scores[selected]
    return row_normalize(graph).astype(np.float32)


def graph_stats(adjacency: np.ndarray, industries: list[str]) -> dict[str, float]:
    non_self = adjacency.copy()
    np.fill_diagonal(non_self, 0.0)
    edge_mask = non_self > 0
    same = np.equal.outer(np.asarray(industries, dtype=object), np.asarray(industries, dtype=object))
    total_weight = float(non_self.sum())
    return {
        "mean_nonself_out_degree": float(edge_mask.sum(axis=1).mean()),
        "maximum_nonself_out_degree": int(edge_mask.sum(axis=1).max(initial=0)),
        "intra_industry_weight_fraction": float(non_self[same].sum() / total_weight) if total_weight else 1.0,
        "self_weight_mean": float(np.diag(adjacency).mean()),
    }


def build(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    guard = StageEDataCustodyGuard.from_config(DEFAULT_CUSTODY, REPO_ROOT)
    panel_path = guard.assert_path_allowed(resolve(config["paths"]["panel"]), purpose="E-3 panel")
    universe_path = guard.assert_path_allowed(resolve(config["paths"]["selected_universe"]), purpose="E-3 universe")
    output_root = guard.assert_path_allowed(resolve(config["paths"]["output_root"]), purpose="E-3 graph outputs")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(universe_path).sort_values("selection_rank", kind="stable")
    stock_codes = universe["stock_code"].astype(str).tolist()
    industries = universe["industry_group"].fillna("其他").astype(str).tolist()
    if len(stock_codes) != len(set(stock_codes)):
        raise ValueError("selected universe contains duplicate stock codes")
    panel = pd.read_csv(panel_path, usecols=["trade_date", "stock_code", config["rolling_correlation_graph"]["return_column"], "is_market_open_week"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    guard.assert_development_frame(panel, date_columns=("trade_date",))
    ceiling = pd.Timestamp(config["development_date_ceiling"])
    if panel["trade_date"].max() > ceiling:
        raise ValueError("panel exceeds E-3 development date ceiling")
    panel_codes = set(panel["stock_code"].astype(str))
    if set(stock_codes) != panel_codes:
        raise ValueError("panel and frozen universe stock sets differ")
    stock_order = universe[["selection_rank", "stock_code", "stock_name", "industry_group"]].copy()
    stock_order["industry_group"] = stock_order["industry_group"].fillna("其他")
    stock_order_path = output_root / "stock_order.csv"
    stock_order.to_csv(stock_order_path, index=False, encoding="utf-8-sig")
    identity = np.eye(len(stock_codes), dtype=np.float32)
    industry = build_industry_graph(industries, config["industry_graph"])
    fixed_path = output_root / "fixed_graphs.npz"
    np.savez_compressed(fixed_path, stock_codes=np.asarray(stock_codes), identity=identity, industry=industry)
    options = config["rolling_correlation_graph"]
    open_dates = sorted(panel.loc[panel["is_market_open_week"].astype(bool), "trade_date"].dropna().unique())
    returns = panel.pivot(index="trade_date", columns="stock_code", values=options["return_column"]).reindex(index=open_dates, columns=stock_codes)
    rolling_graphs = []
    edge_records = []
    stats_records = []
    previous = None
    for position, trade_date in enumerate(returns.index):
        start = max(0, position - int(options["window_weeks"]) + 1)
        window = returns.iloc[start : position + 1]
        if len(window) < int(options["minimum_periods"]):
            adjacency = identity.copy()
        else:
            correlation = window.corr(min_periods=int(options["minimum_periods"])).to_numpy(dtype=float)
            adjacency = correlation_topk(
                correlation, int(options["top_k"]), float(options["minimum_absolute_correlation"]),
                float(options.get("self_weight", 1.0)),
            )
        rolling_graphs.append(adjacency)
        stats = graph_stats(adjacency, industries)
        stats["trade_date"] = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        stats["mean_absolute_change"] = 0.0 if previous is None else float(np.abs(adjacency - previous).mean())
        stats_records.append(stats)
        rows, columns = np.nonzero(adjacency)
        for row, column in zip(rows.tolist(), columns.tolist()):
            edge_records.append({
                "trade_date": stats["trade_date"], "source_stock": stock_codes[row],
                "target_stock": stock_codes[column], "weight": float(adjacency[row, column]),
                "is_self": row == column, "same_industry": industries[row] == industries[column],
            })
        previous = adjacency
    rolling = np.stack(rolling_graphs).astype(np.float32)
    rolling_path = output_root / "rolling_correlation_graphs.npz"
    np.savez_compressed(rolling_path, trade_dates=np.asarray([pd.Timestamp(value).strftime("%Y-%m-%d") for value in returns.index]), stock_codes=np.asarray(stock_codes), adjacency=rolling)
    edges_path = output_root / "rolling_correlation_edges.csv.gz"
    pd.DataFrame(edge_records).to_csv(edges_path, index=False, compression={"method": "gzip", "mtime": 0})
    stats_path = output_root / "graph_stats_by_date.csv"
    pd.DataFrame(stats_records).to_csv(stats_path, index=False)
    summary = {
        "stage": "E-3.1", "graph_batch_id": config["graph_batch_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_date_ceiling": config["development_date_ceiling"],
        "stock_count": len(stock_codes), "date_count": len(returns.index),
        "stock_order_sha256": sha256_file(stock_order_path),
        "fixed_graphs_sha256": sha256_file(fixed_path),
        "rolling_graphs_sha256": sha256_file(rolling_path),
        "rolling_edges_sha256": sha256_file(edges_path),
        "graph_stats_sha256": sha256_file(stats_path),
        "industry_graph_stats": graph_stats(industry, industries),
        "rolling_graph_mean_stats": pd.DataFrame(stats_records).select_dtypes(include="number").mean().to_dict(),
        "future_or_sealed_data_read": False,
        "config_sha256": sha256_file(config_path), "panel_sha256": sha256_file(panel_path),
        "universe_sha256": sha256_file(universe_path),
    }
    summary["batch_sha256"] = stable_json_sha256(summary)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(build(config_path, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
