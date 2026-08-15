from __future__ import annotations

"""Label-free C1 artifact loading and probability-contract audit."""

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


def predict(values: np.ndarray, scale_values: np.ndarray, state: dict[str, np.ndarray], temperature: float, shrinkage: float) -> np.ndarray:
    location = np.clip((values - state["location_median"]) / state["location_scale"], -8.0, 8.0)
    scale = np.clip((scale_values - state["scale_median"]) / state["scale_scale"], -8.0, 8.0)
    location[~np.isfinite(location)] = 0.0; scale[~np.isfinite(scale)] = 0.0
    eta = location @ state["beta"]; sigma = np.exp(np.tanh(scale @ state["gamma"]))
    c0 = 1.0 / (1.0 + np.exp(-(state["thresholds"][0] - eta) / sigma)); c1 = 1.0 / (1.0 + np.exp(-(state["thresholds"][1] - eta) / sigma))
    raw = np.column_stack([c0, c1 - c0, 1.0 - c1]); raw = np.clip(raw, 1e-12, None); raw /= raw.sum(axis=1, keepdims=True)
    softened = np.exp(np.log(raw) / temperature - (np.log(raw) / temperature).max(axis=1, keepdims=True)); softened /= softened.sum(axis=1, keepdims=True)
    result = (1.0 - shrinkage) * softened + shrinkage * state["class_prior"].reshape(1, 3)
    return result / result.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--rg3", required=True, type=Path)
    parser.add_argument("--rg2", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    artifact_path, rg3_path, rg2_path, samples_path, output = args.artifact.resolve(), args.rg3.resolve(), args.rg2.resolve(), args.samples.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8")); model_path = artifact_path.parent / artifact["model_file"]
    if artifact.get("status") != "NON_PRODUCTION_LABEL_FREE_SHADOW_ARTIFACT" or sha256(model_path) != artifact["model_sha256"]:
        raise RuntimeError("artifact identity failure")
    rg3 = pd.read_csv(rg3_path, dtype={"stock_code": str}); rg2 = pd.read_csv(rg2_path, dtype={"stock_code": str})
    # The first two scale fields are frozen RG3 technical features; only the
    # market-wide scale is carried by the sample panel.
    samples = pd.read_csv(samples_path, usecols=["trade_date", "stock_code", "market_volatility_4w"], dtype={"stock_code": str})
    for frame in (rg3, rg2, samples): frame.trade_date = pd.to_datetime(frame.trade_date, errors="raise").dt.normalize()
    location = list(artifact["location_features"]); scale = list(artifact["scale_features"]); state_features = location[14:]
    if (rg2.groupby(["trade_date", "stock_code"], sort=False)[state_features].nunique(dropna=False) > 1).any().any():
        raise RuntimeError("RG2 state differs across repeated decision keys")
    rg2 = rg2.sort_values("sample_key_sha256", kind="mergesort").drop_duplicates(["trade_date", "stock_code"], keep="first")
    scale = list(artifact["scale_features"])
    if (samples.groupby(["trade_date", "stock_code"], sort=False)[["market_volatility_4w"]].nunique(dropna=False) > 1).any().any():
        raise RuntimeError("scale inputs differ across repeated decision keys")
    samples = samples.drop_duplicates(["trade_date", "stock_code"], keep="first")
    joined = rg2[["trade_date", "stock_code", *state_features]].head(300).merge(rg3[["trade_date", "stock_code", *location[:14]]], on=["trade_date", "stock_code"], how="left", validate="many_to_one").merge(samples, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    if joined[location].isna().any().any():
        raise RuntimeError("label-free C1 feature join incomplete")
    if np.isinf(joined[scale].apply(pd.to_numeric, errors="coerce")).any().any():
        raise RuntimeError("non-finite scale feature")
    payload = np.load(model_path, allow_pickle=False)
    state = {name: payload[name] for name in ("beta", "gamma", "thresholds", "location_median", "location_scale", "scale_median", "scale_scale", "class_prior")}
    probability = predict(joined[location].to_numpy(float), joined[scale].to_numpy(float), state, float(artifact["calibration"]["temperature"]), float(artifact["calibration"]["prior_shrinkage"]))
    if probability.shape != (len(joined), 3) or not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("probability contract failure")
    result = {"node_id": "WP22_C1_ARTIFACT_LOAD_AUDIT_V1", "status": "PASS_NON_PRODUCTION_C1_ARTIFACT_LOAD_AND_PROBABILITY_SMOKE", "artifact_sha256": sha256(artifact_path), "model_sha256": sha256(model_path), "rg3_sha256": sha256(rg3_path), "rg2_sha256": sha256(rg2_path), "samples_sha256": sha256(samples_path), "rows": int(len(joined)), "target_labels_read": False, "fresh_labels_read": False, "wp10_outputs_read": False, "metrics_read": False, "gpu_used": False, "production_replacement_allowed": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


