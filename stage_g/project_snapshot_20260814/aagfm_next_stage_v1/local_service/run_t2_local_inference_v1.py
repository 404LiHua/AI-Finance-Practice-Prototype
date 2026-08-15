from __future__ import annotations

"""Offline single-machine inference for the active production T2 model.

This is deliberately a batch CLI: it avoids a network listener until the local
workflow and the candidate-selection gate are both accepted.
"""

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


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def predict(payload: dict, values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dense = payload["dense_model"]
    median = np.asarray(dense["scaler_median"], dtype=float)
    scale = np.asarray(dense["scaler_scale"], dtype=float)
    beta = np.asarray(dense["beta"], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(beta):
        raise ValueError("feature matrix shape mismatch")
    scaled = (values - median) / scale
    scaled[~np.isfinite(scaled)] = 0.0
    scaled = np.clip(scaled, -float(dense["scaler_clip"]), float(dense["scaler_clip"]))
    eta = scaled @ beta
    first, second = (float(item) for item in dense["thresholds"])
    c0 = sigmoid(first - eta); c1 = sigmoid(second - eta)
    raw = np.column_stack([c0, c1 - c0, 1.0 - c1]); raw = np.clip(raw, 1e-12, None); raw /= raw.sum(axis=1, keepdims=True)
    calibration = payload["calibration"]
    temperature = float(calibration["temperature"]); shrinkage = float(calibration["prior_shrinkage"])
    softened = np.exp(np.log(raw) / temperature - (np.log(raw) / temperature).max(axis=1, keepdims=True)); softened /= softened.sum(axis=1, keepdims=True)
    prior = np.asarray(calibration["class_prior"], dtype=float).reshape(1, 3)
    probability = (1.0 - shrinkage) * softened + shrinkage * prior; probability /= probability.sum(axis=1, keepdims=True)
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1)
    confidence = np.clip(1.0 - entropy / np.log(3.0), 0.0, 1.0)
    threshold = float(payload.get("operational_reliability", {}).get("scaled_feature_support_threshold", 4.0))
    excess = np.maximum(np.abs((values - median) / scale) - threshold, 0.0)
    distribution_support = np.exp(-np.mean(excess, axis=1))
    reliability = np.clip(confidence * distribution_support, 0.0, 1.0)
    return probability, {"confidence": confidence, "distribution_support": distribution_support, "reliability": reliability, "gated_ordinal_score": reliability * (probability[:, 2] - probability[:, 0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    registry_path = args.registry.resolve(); input_path = args.input.resolve(); output_path = args.output.resolve(); receipt_path = args.receipt.resolve()
    if output_path.exists() or receipt_path.exists():
        raise RuntimeError("refusing to overwrite inference output or receipt")
    registry = json.loads(registry_path.read_text(encoding="utf-8")); active = registry["active_model"]
    model_path = Path(active["path"])
    if sha256(model_path) != active["sha256"]:
        raise RuntimeError("active model hash mismatch")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if payload["model_id"] != active["model_id"] or payload["target_id"] != "T2_MARKET_RELATIVE_FIXED":
        raise RuntimeError("registry/model contract mismatch")
    if payload["branches"]["graph_weight"] != 0.0 or payload["branches"]["event_weight"] != 0.0:
        raise RuntimeError("active local model must keep graph/event disabled")
    frame = pd.read_parquet(input_path, engine="pyarrow") if input_path.suffix.lower() == ".parquet" else pd.read_csv(input_path, dtype={"stock_code": str})
    features = list(payload["features"])
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing required features: {missing}")
    values = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    probability, diagnostics = predict(payload, values)
    identity = [column for column in ("trade_date", "stock_code") if column in frame.columns]
    result = frame[identity].copy()
    result["model_id"] = active["model_id"]
    result["target_id"] = active["target_id"]
    result["t2_class"] = np.asarray(["DOWN", "NEUTRAL", "UP"], dtype=object)[np.argmax(probability, axis=1)]
    result["prob_down"] = probability[:, 0]; result["prob_neutral"] = probability[:, 1]; result["prob_up"] = probability[:, 2]
    for name, value in diagnostics.items(): result[name] = value
    output_path.parent.mkdir(parents=True, exist_ok=True); receipt_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
    receipt = {"node_id": "AA_GFMNET_LOCAL_T2_INFERENCE_V1", "status": "PASS_LOCAL_CPU_INFERENCE", "registry_sha256": sha256(registry_path), "model_sha256": sha256(model_path), "input_sha256": sha256(input_path), "output_sha256": sha256(output_path), "rows": int(len(result)), "features": features, "gpu_used": False, "fresh_payloads_opened": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "rows": receipt["rows"], "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


