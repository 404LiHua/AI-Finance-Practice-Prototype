from __future__ import annotations

"""Label-free, non-production C1 shadow inference.

The runner deliberately accepts separate RG3, RG2 and scale panels, reads only
the declared feature columns, and writes an auditable prediction receipt.  It
never opens target columns, FRESH payloads, the production registry, or a GPU.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BAN = ("fresh", "screening", "final", "sealed_holdout")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scale_panel(path: Path) -> pd.DataFrame:
    """Read CSV/CSV.GZ/Parquet and require a stock-keyed scale panel."""
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, dtype={"stock_code": str})
    except Exception as exc:
        raise RuntimeError(f"scale panel cannot be read as CSV/Parquet: {exc}") from exc
    required = {"trade_date", "stock_code", "market_volatility_4w"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"scale panel schema missing required keyed columns: {missing}")
    return frame[["trade_date", "stock_code", "market_volatility_4w"]].copy()


def predict(values: np.ndarray, scale_values: np.ndarray, state: dict[str, np.ndarray], temperature: float, shrinkage: float) -> np.ndarray:
    location = (values - state["location_median"]) / state["location_scale"]
    scale = (scale_values - state["scale_median"]) / state["scale_scale"]
    location[~np.isfinite(location)] = 0.0; scale[~np.isfinite(scale)] = 0.0
    location = np.clip(location, -8.0, 8.0); scale = np.clip(scale, -8.0, 8.0)
    eta = location @ state["beta"]; sigma = np.exp(np.tanh(scale @ state["gamma"]))
    first, second = state["thresholds"]
    c0 = 1.0 / (1.0 + np.exp(-(first - eta) / sigma)); c1 = 1.0 / (1.0 + np.exp(-(second - eta) / sigma))
    raw = np.column_stack([c0, c1 - c0, 1.0 - c1]); raw = np.clip(raw, 1e-12, None); raw /= raw.sum(axis=1, keepdims=True)
    log_raw = np.log(raw) / float(temperature); softened = np.exp(log_raw - log_raw.max(axis=1, keepdims=True)); softened /= softened.sum(axis=1, keepdims=True)
    return ((1.0 - shrinkage) * softened + shrinkage * state["class_prior"].reshape(1, 3)), location, scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--rg3", required=True, type=Path)
    parser.add_argument("--rg2", required=True, type=Path)
    parser.add_argument("--scale-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    artifact_path, rg3_path, rg2_path, scale_path = (args.artifact.resolve(), args.rg3.resolve(), args.rg2.resolve(), args.scale_panel.resolve())
    output, receipt_path = args.output.resolve(), args.receipt.resolve()
    if output.exists() or receipt_path.exists():
        raise RuntimeError("refusing to overwrite output or receipt")
    if any(token in str(path).lower() for path in (artifact_path, rg3_path, rg2_path, scale_path, output, receipt_path) for token in BAN):
        raise RuntimeError("prohibited path token")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("status") != "NON_PRODUCTION_LABEL_FREE_SHADOW_ARTIFACT" or artifact.get("candidate_id") != "REV8_C1_RG2_STATE_AUGMENTED_HETEROSKEDASTIC_ORDINAL":
        raise RuntimeError("C1 artifact status/identity failure")
    model_path = artifact_path.parent / artifact["model_file"]
    if sha256(model_path) != artifact["model_sha256"]:
        raise RuntimeError("C1 model hash mismatch")
    rg3 = pd.read_csv(rg3_path, dtype={"stock_code": str})
    rg2 = pd.read_csv(rg2_path, dtype={"stock_code": str})
    scale = read_scale_panel(scale_path)
    for frame in (rg3, rg2, scale):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    location = list(artifact["location_features"]); rg3_features = location[:14]; state_features = location[14:]; scale_features = list(artifact["scale_features"])
    required_rg3 = {"trade_date", "stock_code", *rg3_features}; required_rg2 = {"trade_date", "stock_code", *state_features}
    if not required_rg3.issubset(rg3.columns) or not required_rg2.issubset(rg2.columns):
        raise RuntimeError("candidate feature columns missing")
    if rg3.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("RG3 candidate key duplication")
    if (rg2.groupby(["trade_date", "stock_code"], sort=False)[state_features].nunique(dropna=False) > 1).any().any():
        raise RuntimeError("RG2 candidate state differs across repeated fold copies")
    rg2 = rg2.sort_values(["trade_date", "stock_code"], kind="mergesort").drop_duplicates(["trade_date", "stock_code"], keep="first")
    if (scale.groupby(["trade_date", "stock_code"], sort=False)[["market_volatility_4w"]].nunique(dropna=False) > 1).any().any():
        raise RuntimeError("scale panel differs across repeated fold copies")
    scale = scale.drop_duplicates(["trade_date", "stock_code"], keep="first")
    keys = rg2[["trade_date", "stock_code", *state_features]]
    joined = keys.merge(rg3[["trade_date", "stock_code", *rg3_features]], on=["trade_date", "stock_code"], how="inner", validate="one_to_one").merge(scale, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        joined = joined.head(args.limit).copy()
    if joined.empty or joined[[*location, "market_volatility_4w"]].isna().all(axis=1).any():
        raise RuntimeError("candidate feature join is empty or entirely missing on a row")
    if np.isinf(joined[[*location, "market_volatility_4w"]].apply(pd.to_numeric, errors="coerce")).any().any():
        raise RuntimeError("candidate feature input contains infinity")
    state_npz = np.load(model_path, allow_pickle=False)
    model_state = {name: state_npz[name] for name in ("beta", "gamma", "thresholds", "location_median", "location_scale", "scale_median", "scale_scale", "class_prior")}
    raw_probability, scaled_location, scaled_scale = predict(joined[location].to_numpy(float), joined[scale_features].to_numpy(float), model_state, float(artifact["calibration"]["temperature"]), float(artifact["calibration"]["prior_shrinkage"]))
    probability = raw_probability / raw_probability.sum(axis=1, keepdims=True)
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1)
    confidence = np.clip(1.0 - entropy / np.log(3.0), 0.0, 1.0)
    support = np.exp(-np.mean(np.maximum(np.abs(np.column_stack([scaled_location, scaled_scale])) - 4.0, 0.0), axis=1))
    reliability = np.clip(confidence * support, 0.0, 1.0)
    result = joined[["trade_date", "stock_code"]].copy()
    result["candidate_id"] = artifact["candidate_id"]; result["target_id"] = artifact["target_id"]
    result["t2_class"] = np.asarray(["DOWN", "NEUTRAL", "UP"], dtype=object)[np.argmax(probability, axis=1)]
    result["prob_down"] = probability[:, 0]; result["prob_neutral"] = probability[:, 1]; result["prob_up"] = probability[:, 2]
    result["confidence"] = confidence; result["distribution_support"] = support; result["reliability"] = reliability; result["gated_ordinal_score"] = reliability * (probability[:, 2] - probability[:, 0])
    if not np.isfinite(result[["prob_down", "prob_neutral", "prob_up", "confidence", "distribution_support", "reliability", "gated_ordinal_score"]].to_numpy(float)).all() or not np.allclose(result[["prob_down", "prob_neutral", "prob_up"]].sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("candidate output probability/reliability contract failure")
    output.parent.mkdir(parents=True, exist_ok=True); receipt_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, engine="pyarrow", compression="zstd")
    receipt = {"node_id": "WP22_C1_LABEL_FREE_SHADOW_INFERENCE_V1", "status": "PASS_NON_PRODUCTION_C1_LABEL_FREE_SHADOW_INFERENCE", "artifact_sha256": sha256(artifact_path), "model_sha256": sha256(model_path), "rg3_sha256": sha256(rg3_path), "rg2_sha256": sha256(rg2_path), "scale_panel_sha256": sha256(scale_path), "output_sha256": sha256(output), "rows": int(len(result)), "target_columns_read": False, "fresh_payloads_opened": False, "wp10_outputs_read": False, "metrics_read": False, "gpu_used": False, "production_registry_modified": False, "automatic_trading": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "rows": receipt["rows"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

