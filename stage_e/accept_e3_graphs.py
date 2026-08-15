"""Machine-verifiable acceptance for Stage E-3 graph artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def accept(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    paths = {
        "stock_order": root / "stock_order.csv",
        "fixed": root / "fixed_graphs.npz",
        "rolling": root / "rolling_correlation_graphs.npz",
        "edges": root / "rolling_correlation_edges.csv.gz",
        "stats": root / "graph_stats_by_date.csv",
        "summary": root / "summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    result = {"stage": "E-3.1", "graph_batch_id": config["graph_batch_id"], "passed": False, "missing": missing}
    if missing:
        return result
    stock_order = pd.read_csv(paths["stock_order"])
    stock_codes = stock_order["stock_code"].astype(str).tolist()
    fixed = np.load(paths["fixed"])
    rolling = np.load(paths["rolling"])
    identity = fixed["identity"]
    industry = fixed["industry"]
    adjacency = rolling["adjacency"]
    dates = pd.to_datetime(rolling["trade_dates"].astype(str), errors="coerce")
    declared_fixed_codes = fixed["stock_codes"].astype(str).tolist()
    declared_rolling_codes = rolling["stock_codes"].astype(str).tolist()
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    batch_sha = summary.get("batch_sha256", "")
    summary_payload = dict(summary)
    summary_payload.pop("batch_sha256", None)
    expected_hashes = {
        "stock_order_sha256": paths["stock_order"], "fixed_graphs_sha256": paths["fixed"],
        "rolling_graphs_sha256": paths["rolling"], "rolling_edges_sha256": paths["edges"],
        "graph_stats_sha256": paths["stats"],
    }
    top_k = int(config["rolling_correlation_graph"]["top_k"])
    minimum_periods = int(config["rolling_correlation_graph"]["minimum_periods"])
    checks = {
        "stock_order_unique": len(stock_codes) == len(set(stock_codes)),
        "stock_order_consistent": stock_codes == declared_fixed_codes == declared_rolling_codes,
        "identity_shape": identity.shape == (len(stock_codes), len(stock_codes)),
        "industry_shape": industry.shape == identity.shape,
        "rolling_shape": adjacency.shape == (len(dates), len(stock_codes), len(stock_codes)),
        "dates_parseable": bool(dates.notna().all()),
        "development_ceiling_respected": bool(dates.max() <= pd.Timestamp(config["development_date_ceiling"])),
        "finite_nonnegative": bool(np.isfinite(industry).all() and np.isfinite(adjacency).all() and (industry >= 0).all() and (adjacency >= 0).all()),
        "identity_exact": bool(np.array_equal(identity, np.eye(len(stock_codes), dtype=np.float32))),
        "industry_row_stochastic": bool(np.allclose(industry.sum(axis=-1), 1.0, atol=1e-6)),
        "rolling_row_stochastic": bool(np.allclose(adjacency.sum(axis=-1), 1.0, atol=1e-6)),
        "rolling_topk_bounded": bool(((adjacency > 0).sum(axis=-1) <= top_k + 1).all()),
        "warmup_is_identity": bool(np.array_equal(adjacency[: max(0, minimum_periods - 1)], np.broadcast_to(identity, adjacency[: max(0, minimum_periods - 1)].shape))),
        "artifact_hashes_valid": all(summary.get(key) == sha256_file(path) for key, path in expected_hashes.items()),
        "batch_sha_valid": batch_sha == stable_json_sha256(summary_payload),
        "future_or_sealed_data_not_read": not summary.get("future_or_sealed_data_read", True),
    }
    result.update({
        "passed": all(checks.values()), "checks": checks, "stock_count": len(stock_codes),
        "date_count": len(dates), "maximum_date": dates.max().strftime("%Y-%m-%d"),
        "rolling_graph_shape": list(adjacency.shape), "batch_sha256": batch_sha,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e3_graph_acceptance_v1.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    result = accept(config_path)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), **result}
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
