from __future__ import annotations

"""Validate and seal the two-model, label-free WP11 shadow prediction set."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["trade_date", "stock_code"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_probability(frame: pd.DataFrame, model_id: str) -> None:
    if frame.duplicated(KEY).any() or frame.model_id.nunique() != 1 or frame.model_id.iloc[0] != model_id:
        raise RuntimeError(f"model/key contract failure: {model_id}")
    values = frame[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError(f"probability contract failure: {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--market-state", required=True, type=Path)
    parser.add_argument("--incumbent-predictions", required=True, type=Path)
    parser.add_argument("--c0-predictions", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    feature_path, market_path, incumbent_path, c0_path, protocol_path, output = (args.features.resolve(), args.market_state.resolve(), args.incumbent_predictions.resolve(), args.c0_predictions.resolve(), args.protocol.resolve(), args.output_root.resolve())
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_C0_ARTIFACT_MATERIALIZATION_OR_WP11_SHADOW_PREDICTION":
        raise RuntimeError("WP11 protocol is not frozen")
    ids = protocol["input_identities"]
    if sha256(feature_path) != ids["label_free_feature_sha256"] or sha256(incumbent_path) != ids["incumbent_shadow_prediction_sha256"]:
        raise RuntimeError("sealed WP11 input identity mismatch")
    features = pd.read_parquet(feature_path, engine="pyarrow"); market = pd.read_parquet(market_path, engine="pyarrow")
    incumbent = pd.read_parquet(incumbent_path, engine="pyarrow"); c0 = pd.read_parquet(c0_path, engine="pyarrow")
    for frame in (features, market, incumbent, c0):
        if "trade_date" in frame.columns:
            frame.trade_date = pd.to_datetime(frame.trade_date, errors="raise").dt.normalize()
    origin = pd.Timestamp(protocol["shadow"]["origin_date"]).normalize()
    if len(features) != int(protocol["shadow"]["frozen_universe_size"]) or not features.trade_date.eq(origin).all() or len(market) != 1 or market.trade_date.iloc[0] != origin:
        raise RuntimeError("WP11 origin/input cardinality contract failure")
    validate_probability(incumbent, protocol["governance"]["active_production_model"])
    validate_probability(c0, protocol["governance"]["candidate_id"])
    expected = features[KEY].sort_values(KEY, kind="mergesort").reset_index(drop=True)
    for label, frame in (("incumbent", incumbent), ("c0", c0)):
        actual = frame[KEY].sort_values(KEY, kind="mergesort").reset_index(drop=True)
        if not expected.equals(actual):
            raise RuntimeError(f"WP11 key coverage mismatch: {label}")
    output.mkdir(parents=True)
    manifest = {"node_id": "WP11_LABEL_FREE_TWO_MODEL_PREDICTION_SEAL_V1", "status": "SEALED_LABEL_FREE_SHADOW_PENDING_FUTURE_INDEPENDENT_AUTHORIZATION", "protocol_sha256": sha256(protocol_path), "origin_date": origin.date().isoformat(), "rows": int(len(features)), "input_sha256": {"features": sha256(feature_path), "market_state": sha256(market_path), "incumbent_predictions": sha256(incumbent_path), "c0_predictions": sha256(c0_path)}, "models": [protocol["governance"]["active_production_model"], protocol["governance"]["candidate_id"]], "target_labels_read": False, "fresh_labels_read": False, "wp10_outputs_read": False, "metrics_read": False, "model_trained": False, "gpu_used": False, "automatic_trading": False, "production_replacement_allowed": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path = output / "WP11_LABEL_FREE_SHADOW_PREDICTION_SEAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {"node_id": manifest["node_id"], "status": manifest["status"], "manifest_sha256": sha256(manifest_path), "target_labels_read": False, "metrics_read": False, "gpu_used": False}
    (output / "WP11_SHADOW_SEAL_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest_sha256": receipt["manifest_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


