from __future__ import annotations

"""Materialize the fixed WP09 C0 definition as a reusable CPU inference artifact.

The only labels opened are the already-authorized development labels contained
in the frozen source package.  No WP10/FRESH payload, metric, or prediction is
opened by this program.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path, root: Path):
    sys.dont_write_bytecode = True
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mcc(target: np.ndarray, prediction: np.ndarray) -> float:
    matrix = np.zeros((3, 3), dtype=float)
    np.add.at(matrix, (np.asarray(target, dtype=int), np.asarray(prediction, dtype=int)), 1)
    total = matrix.sum(); truth = matrix.sum(axis=1); pred = matrix.sum(axis=0)
    denominator = np.sqrt((total * total - np.dot(truth, truth)) * (total * total - np.dot(pred, pred)))
    return float((np.trace(matrix) * total - np.dot(truth, pred)) / denominator) if denominator else 0.0


def select_calibration(dev: pd.DataFrame, xcols: list[str], zcols: list[str], hetero, ordinal) -> dict:
    splits = ordinal.expanding_time_calibration_splits(dev.trade_date.to_numpy(), folds=3, embargo_weeks=11, minimum_fit_weeks=26)
    probs: list[np.ndarray] = []; targets: list[np.ndarray] = []; priors: list[np.ndarray] = []
    for fit_mask, cal_mask in splits:
        fit, cal = dev.loc[fit_mask], dev.loc[cal_mask]
        fitted = hetero.fit_heteroscedastic_proportional_odds(fit[xcols].to_numpy(float), fit[zcols].to_numpy(float), fit.ordinal_target.astype(int).to_numpy(), location_l2=0.001, scale_l2=0.01, max_iter=200)
        probs.append(fitted.predict_proba(cal[xcols].to_numpy(float), cal[zcols].to_numpy(float)))
        targets.append(cal.ordinal_target.astype(int).to_numpy())
        prior = np.bincount(fit.ordinal_target.astype(int), minlength=3).astype(float); prior /= prior.sum()
        priors.append(np.repeat(prior.reshape(1, 3), len(cal), axis=0))
    probability = np.vstack(probs); target = np.concatenate(targets); prior = np.vstack(priors)
    grid: list[dict] = []
    for temperature in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        for shrinkage in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5):
            calibrated = ordinal.temperature_and_prior_calibrate(probability, prior, temperature, shrinkage)
            brier = float(np.mean(np.sum((calibrated - np.eye(3)[target]) ** 2, axis=1)))
            grid.append({"temperature": temperature, "prior_shrinkage": shrinkage, "brier": brier, "mcc": mcc(target, np.argmax(calibrated, axis=1))})
    lowest_brier = min(row["brier"] for row in grid)
    admissible = [row for row in grid if row["brier"] <= lowest_brier + 0.005]
    chosen = min(admissible, key=lambda row: (-row["mcc"], row["temperature"], row["prior_shrinkage"]))
    return {"temperature": chosen["temperature"], "prior_shrinkage": chosen["prior_shrinkage"], "inner_oof_rows": int(len(target)), "inner_split_count": int(len(splits))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--feature-input", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source, feature_path, protocol_path, output = args.source_root.resolve(), args.feature_input.resolve(), args.protocol.resolve(), args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); identities = protocol.get("input_identities", {})
    if protocol.get("status") != "FROZEN_BEFORE_C0_ARTIFACT_MATERIALIZATION_OR_WP11_SHADOW_PREDICTION" or not protocol.get("governance", {}).get("artifact_materialization_allowed"):
        raise RuntimeError("WP11 protocol does not authorize fixed C0 artifact materialization")
    samples_path = source / "data" / "rg1_4_materialized" / "samples.csv.gz"
    source_paths = {
        "source_rev8_targets_sha256": source / "src" / "rev8_targets.py",
        "source_rg3_features_sha256": source / "src" / "rg3_features.py",
        "source_rg2_ordinal_sha256": source / "src" / "rg2_calibrated_ordinal.py",
        "c0_source_sha256": Path(__file__).resolve().parents[1] / "src" / "heteroscedastic_ordinal_t2_v1.py",
    }
    if sha256(samples_path) != identities["development_samples_sha256"] or sha256(feature_path) != identities["development_rg3_features_sha256"]:
        raise RuntimeError("frozen development data identity mismatch")
    for key, path in source_paths.items():
        if not path.is_file() or sha256(path) != identities[key]:
            raise RuntimeError(f"frozen source identity mismatch: {key}")
    os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    target = load_module("wp11_target", source_paths["source_rev8_targets_sha256"], source)
    rg3 = load_module("wp11_rg3", source_paths["source_rg3_features_sha256"], source)
    ordinal = load_module("wp11_ordinal", source_paths["source_rg2_ordinal_sha256"], source)
    hetero = load_module("wp11_hetero", source_paths["c0_source_sha256"], Path(__file__).resolve().parents[1])
    sample_columns = ["trade_date", "stock_code", "target_return_h4", "target_valid", "realized_volatility_8w", "market_volatility_4w"]
    samples = pd.read_csv(samples_path, usecols=sample_columns, dtype={"stock_code": str})
    samples.trade_date = pd.to_datetime(samples.trade_date, errors="raise").dt.normalize()
    canonical = samples.drop_duplicates(["trade_date", "stock_code"], keep="first")
    if len(canonical) == 0 or canonical.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("development sample key contract failure")
    variants = target.build_target_variants(canonical[["trade_date", "stock_code", "target_return_h4", "target_valid", "realized_volatility_8w"]])
    labels = variants[["trade_date", "stock_code", "T2_valid", "T2_label"]].rename(columns={"T2_valid": "target_valid", "T2_label": "ordinal_target"})
    xcols = list(rg3.DAILY_TECHNICAL_FEATURES); zcols = ["realized_volatility_20d", "downside_volatility_60d", "market_volatility_4w"]
    features = pd.read_csv(feature_path, usecols=["trade_date", "stock_code", *xcols], dtype={"stock_code": str})
    features.trade_date = pd.to_datetime(features.trade_date, errors="raise").dt.normalize()
    dev = canonical[["trade_date", "stock_code", "market_volatility_4w"]].merge(labels, on=["trade_date", "stock_code"], how="left", validate="one_to_one").merge(features, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    dev = dev[dev.target_valid.astype(bool)].copy()
    if dev.empty or dev[xcols].isna().any().any() or dev.duplicated(["trade_date", "stock_code"]).any() or np.isinf(dev[zcols].to_numpy(float)).any():
        raise RuntimeError("C0 development contract failure")
    calibration = select_calibration(dev, xcols, zcols, hetero, ordinal)
    fitted = hetero.fit_heteroscedastic_proportional_odds(dev[xcols].to_numpy(float), dev[zcols].to_numpy(float), dev.ordinal_target.astype(int).to_numpy(), location_l2=0.001, scale_l2=0.01, max_iter=200)
    prior = np.bincount(dev.ordinal_target.astype(int), minlength=3).astype(float); prior /= prior.sum()
    artifact = {
        "artifact_id": "WP11_C0_FIXED_INFERENCE_ARTIFACT_V1",
        "status": "FIXED_C0_ARTIFACT_NOT_A_PRODUCTION_MODEL",
        "candidate_id": protocol["governance"]["candidate_id"],
        "target_id": "T2_MARKET_RELATIVE_FIXED",
        "location_features": xcols,
        "scale_features": zcols,
        "beta": fitted.beta.tolist(), "gamma": fitted.gamma.tolist(), "thresholds": [float(v) for v in fitted.thresholds],
        "location_scaler": {"median": fitted.location_scaler.median.tolist(), "scale": fitted.location_scaler.scale.tolist(), "clip": fitted.location_scaler.clip},
        "scale_scaler": {"median": fitted.scale_scaler.median.tolist(), "scale": fitted.scale_scaler.scale.tolist(), "clip": fitted.scale_scaler.clip},
        "calibration": {**calibration, "class_prior": prior.tolist()},
        "fixed_parameters": {"location_l2": 0.001, "scale_l2": 0.01, "log_scale_bound": 1.0, "max_iter": 200, "optimizer": "torch_LBFGS_float64_strong_wolfe"},
        "input_sha256": {"samples": sha256(samples_path), "rg3_features": sha256(feature_path), "protocol": sha256(protocol_path), **{key: sha256(path) for key, path in source_paths.items()}},
        "development_rows": int(len(dev)),
        "target_labels_read": "authorized_development_only",
        "fresh_labels_read": False,
        "wp10_outputs_read": False,
        "metrics_read": False,
        "gpu_used": False,
        "production_replacement_allowed": False,
    }
    output.mkdir(parents=True)
    artifact_path = output / "WP11_C0_FIXED_INFERENCE_ARTIFACT.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {"node_id": "AA_GFMNET_WP11_C0_FIXED_ARTIFACT_V1", "status": "PASS_FIXED_C0_ARTIFACT_MATERIALIZED_DEVELOPMENT_ONLY", "protocol_sha256": sha256(protocol_path), "artifact_sha256": sha256(artifact_path), "development_rows": int(len(dev)), "target_labels_read": "authorized_development_only", "fresh_labels_read": False, "wp10_outputs_read": False, "metrics_read": False, "model_selection_performed": False, "gpu_used": False, "production_assets_modified": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "WP11_C0_ARTIFACT_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "artifact_sha256": receipt["artifact_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


