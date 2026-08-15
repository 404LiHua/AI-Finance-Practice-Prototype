from __future__ import annotations

"""Run the fixed non-production C0 artifact on one label-free WP11 input."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_transform(values: np.ndarray, scaler: dict) -> np.ndarray:
    median = np.asarray(scaler["median"], dtype=float); scale = np.asarray(scaler["scale"], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(median) or len(scale) != len(median) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("C0 artifact scaler contract failure")
    transformed = (values - median) / scale
    transformed[~np.isfinite(transformed)] = 0.0
    return np.clip(transformed, -float(scaler["clip"]), float(scaler["clip"]))


def predict(artifact: dict, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = robust_transform(location, artifact["location_scaler"]); z = robust_transform(scale, artifact["scale_scaler"])
    beta = np.asarray(artifact["beta"], dtype=float); gamma = np.asarray(artifact["gamma"], dtype=float); thresholds = np.asarray(artifact["thresholds"], dtype=float)
    if x.shape[1] != len(beta) or z.shape[1] != len(gamma) or thresholds.shape != (2,) or thresholds[1] <= thresholds[0]:
        raise RuntimeError("C0 artifact parameter contract failure")
    eta = x @ beta; sigma = np.exp(np.tanh(z @ gamma))
    c0 = 1.0 / (1.0 + np.exp(-np.clip((thresholds[0] - eta) / sigma, -60.0, 60.0)))
    c1 = 1.0 / (1.0 + np.exp(-np.clip((thresholds[1] - eta) / sigma, -60.0, 60.0)))
    raw = np.column_stack([c0, c1 - c0, 1.0 - c1]); raw = np.clip(raw, 1e-12, None); raw /= raw.sum(axis=1, keepdims=True)
    cal = artifact["calibration"]; temperature = float(cal["temperature"]); shrinkage = float(cal["prior_shrinkage"])
    softened = np.exp(np.log(raw) / temperature - (np.log(raw) / temperature).max(axis=1, keepdims=True)); softened /= softened.sum(axis=1, keepdims=True)
    prior = np.asarray(cal["class_prior"], dtype=float).reshape(1, 3)
    probability = (1.0 - shrinkage) * softened + shrinkage * prior
    return probability / probability.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--market-state", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    features_path, market_path, artifact_path, protocol_path, output_path, receipt_path = (args.features.resolve(), args.market_state.resolve(), args.artifact.resolve(), args.protocol.resolve(), args.output.resolve(), args.receipt.resolve())
    if output_path.exists() or receipt_path.exists():
        raise RuntimeError("refusing to overwrite WP11 C0 shadow output")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_C0_ARTIFACT_MATERIALIZATION_OR_WP11_SHADOW_PREDICTION":
        raise RuntimeError("WP11 protocol is not frozen")
    if sha256(features_path) != protocol["input_identities"]["label_free_feature_sha256"]:
        raise RuntimeError("WP11 feature identity mismatch")
    if artifact.get("artifact_id") != "WP11_C0_FIXED_INFERENCE_ARTIFACT_V1" or artifact.get("candidate_id") != protocol["governance"]["candidate_id"] or artifact.get("target_id") != "T2_MARKET_RELATIVE_FIXED":
        raise RuntimeError("C0 artifact/protocol contract failure")
    features = pd.read_parquet(features_path, engine="pyarrow"); market = pd.read_parquet(market_path, engine="pyarrow")
    forbidden = set(protocol["shadow"]["forbidden_payload_columns"])
    if forbidden & set(features.columns) or forbidden & set(market.columns):
        raise RuntimeError("label-bearing column found in purported label-free WP11 input")
    features.trade_date = pd.to_datetime(features.trade_date, errors="raise").dt.normalize(); features.source_trade_date = pd.to_datetime(features.source_trade_date, errors="raise").dt.normalize()
    market.trade_date = pd.to_datetime(market.trade_date, errors="raise").dt.normalize()
    origin = pd.Timestamp(protocol["shadow"]["origin_date"]).normalize()
    xcols = list(artifact["location_features"]); zcols = list(artifact["scale_features"])
    if features.duplicated(["trade_date", "stock_code"]).any() or len(features) != int(protocol["shadow"]["frozen_universe_size"]) or not features.trade_date.eq(origin).all():
        raise RuntimeError("WP11 feature key/date contract failure")
    if len(market) != 1 or market.trade_date.iloc[0] != origin or not np.isfinite(pd.to_numeric(market.market_volatility_4w, errors="coerce")).all():
        raise RuntimeError("WP11 market-state contract failure")
    if set(xcols).difference(features.columns) or set(zcols[:2]).difference(features.columns):
        raise RuntimeError("WP11 C0 feature column missing")
    location = features[xcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    scale = np.column_stack([features[zcols[0]].to_numpy(float), features[zcols[1]].to_numpy(float), np.repeat(float(market.market_volatility_4w.iloc[0]), len(features))])
    probability = predict(artifact, location, scale)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("C0 probability contract failure")
    result = features[["trade_date", "stock_code", "source_trade_date"]].copy()
    result["model_id"] = artifact["candidate_id"]; result["target_id"] = artifact["target_id"]
    result["t2_class"] = np.asarray(["DOWN", "NEUTRAL", "UP"], dtype=object)[np.argmax(probability, axis=1)]
    result["prob_down"] = probability[:, 0]; result["prob_neutral"] = probability[:, 1]; result["prob_up"] = probability[:, 2]
    result["candidate_ordinal_score"] = probability[:, 2] - probability[:, 0]
    output_path.parent.mkdir(parents=True, exist_ok=True); receipt_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
    receipt = {"node_id": "AA_GFMNET_WP11_C0_LABEL_FREE_SHADOW_INFERENCE_V1", "status": "PASS_C0_LABEL_FREE_WP11_SHADOW_INFERENCE", "protocol_sha256": sha256(protocol_path), "artifact_sha256": sha256(artifact_path), "feature_sha256": sha256(features_path), "market_state_sha256": sha256(market_path), "output_sha256": sha256(output_path), "rows": int(len(result)), "target_labels_read": False, "fresh_labels_read": False, "wp10_outputs_read": False, "metrics_read": False, "model_trained": False, "gpu_used": False, "automatic_trading": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "rows": receipt["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


