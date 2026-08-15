from __future__ import annotations

"""Build the frozen C1 full-development, non-production shadow artifact.

The program only consumes already-authorized development labels.  It produces
no development metrics and cannot read a FRESH/holdout path.
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


BAN = ("fresh", "screening", "final", "sealed_holdout")
TEMPERATURES = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
SHRINKAGES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)


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
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mcc(target: np.ndarray, prediction: np.ndarray) -> float:
    table = np.zeros((3, 3), dtype=float)
    np.add.at(table, (target.astype(int), prediction.astype(int)), 1)
    total = table.sum(); truth = table.sum(1); predicted = table.sum(0)
    denominator = np.sqrt((total ** 2 - np.dot(truth, truth)) * (total ** 2 - np.dot(predicted, predicted)))
    return float((np.trace(table) * total - np.dot(truth, predicted)) / denominator) if denominator else 0.0


def brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.sum((probability - np.eye(3)[target.astype(int)]) ** 2, axis=1)))


def calibrate(frame: pd.DataFrame, location: list[str], scale: list[str], hetero, ordinal) -> dict:
    splits = ordinal.expanding_time_calibration_splits(
        frame.trade_date.to_numpy(), folds=3, embargo_weeks=11, minimum_fit_weeks=26
    )
    probabilities, targets, priors = [], [], []
    for fit_mask, calibration_mask in splits:
        fit = frame.loc[fit_mask]; calibration = frame.loc[calibration_mask]
        model = hetero.fit_heteroscedastic_proportional_odds(
            fit[location].to_numpy(float), fit[scale].to_numpy(float), fit.ordinal_target.astype(int).to_numpy(),
            location_l2=0.001, scale_l2=0.01, max_iter=200,
        )
        probabilities.append(model.predict_proba(calibration[location].to_numpy(float), calibration[scale].to_numpy(float)))
        targets.append(calibration.ordinal_target.astype(int).to_numpy())
        prior = np.bincount(fit.ordinal_target.astype(int), minlength=3).astype(float); prior /= prior.sum()
        priors.append(np.repeat(prior.reshape(1, 3), len(calibration), axis=0))
    probability = np.vstack(probabilities); target = np.concatenate(targets); prior = np.vstack(priors)
    choices = []
    for temperature in TEMPERATURES:
        for shrinkage in SHRINKAGES:
            calibrated = ordinal.temperature_and_prior_calibrate(probability, prior, temperature, shrinkage)
            choices.append({"temperature": temperature, "prior_shrinkage": shrinkage, "brier": brier(target, calibrated), "mcc": mcc(target, np.argmax(calibrated, axis=1))})
    minimum = min(choice["brier"] for choice in choices)
    admissible = [choice for choice in choices if choice["brier"] <= minimum + 0.005]
    return min(admissible, key=lambda choice: (-choice["mcc"], choice["temperature"], choice["prior_shrinkage"])) | {"inner_oof_rows": int(len(target)), "inner_split_count": len(splits)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--rg3-input", required=True, type=Path)
    parser.add_argument("--rg2-input", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source, rg3_path, rg2_path, protocol_path, output = (args.source_root.resolve(), args.rg3_input.resolve(), args.rg2_input.resolve(), args.protocol.resolve(), args.output_root.resolve())
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    if any(token in str(path).lower() for path in (source, rg3_path, rg2_path, protocol_path, output) for token in BAN):
        raise RuntimeError("prohibited path token")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_C1_FULL_REFIT_OR_FUTURE_INPUT_OR_LABEL_READ":
        raise RuntimeError("WP22 protocol is not frozen")
    expected = protocol["immutable_development_inputs"]
    samples_path = source / "data" / "rg1_4_materialized" / "samples.csv.gz"
    weekly_path = source / "data" / "rg1_4_materialized" / "weekly_panel.csv.gz"
    hetero_path = Path(__file__).resolve().parents[1] / "src" / "heteroscedastic_ordinal_t2_v1.py"
    for label, path, digest in (("samples", samples_path, expected["samples_sha256"]), ("weekly", weekly_path, expected["weekly_panel_sha256"]), ("rg3", rg3_path, expected["rg3_features_sha256"]), ("rg2", rg2_path, expected["rg2_state_features_sha256"]), ("hetero", hetero_path, expected["heteroscedastic_source_sha256"])):
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"{label} hash mismatch")
    os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    target = load_module("wp22_target", source / "src" / "rev8_targets.py", source)
    rg3 = load_module("wp22_rg3", source / "src" / "rg3_features.py", source)
    ordinal = load_module("wp22_ordinal", source / "src" / "rg2_calibrated_ordinal.py", source)
    hetero = load_module("wp22_hetero", hetero_path, hetero_path.parent)
    usecols = ["trade_date", "stock_code", "target_return_h4", "target_valid", "realized_volatility_8w", "market_volatility_4w"]
    samples = pd.read_csv(samples_path, usecols=usecols, dtype={"stock_code": str})
    samples.trade_date = pd.to_datetime(samples.trade_date, errors="raise").dt.normalize()
    if (samples.groupby(["trade_date", "stock_code"], sort=False)[["target_return_h4", "target_valid", "realized_volatility_8w", "market_volatility_4w"]].nunique(dropna=False) > 1).any().any():
        raise RuntimeError("noncanonical full-development sample values")
    canonical = samples.drop_duplicates(["trade_date", "stock_code"], keep="first")
    variants = target.build_target_variants(canonical[["trade_date", "stock_code", "target_return_h4", "target_valid", "realized_volatility_8w"]])
    labels = variants[["trade_date", "stock_code", "T2_valid", "T2_label"]].rename(columns={"T2_valid": "target_valid", "T2_label": "ordinal_target"})
    rg3_frame = pd.read_csv(rg3_path, usecols=["trade_date", "stock_code", "source_trade_date", *rg3.DAILY_TECHNICAL_FEATURES], dtype={"stock_code": str})
    rg3_frame.trade_date = pd.to_datetime(rg3_frame.trade_date, errors="raise").dt.normalize(); rg3_frame.source_trade_date = pd.to_datetime(rg3_frame.source_trade_date, errors="raise").dt.normalize()
    if rg3_frame.duplicated(["trade_date", "stock_code"]).any() or (rg3_frame.source_trade_date > rg3_frame.trade_date).any() or rg3_frame[rg3.DAILY_TECHNICAL_FEATURES].isna().any().any():
        raise RuntimeError("RG3 PIT contract failure")
    state = pd.read_csv(rg2_path, dtype={"stock_code": str})
    state.trade_date = pd.to_datetime(state.trade_date, errors="raise").dt.normalize()
    state_features = [column for column in state.columns if column not in {"sample_key_sha256", "fold_id", "split_role", "trade_date", "stock_code"}]
    if len(state_features) != 18 or state.duplicated("sample_key_sha256").any() or state[state_features].isna().any().any():
        raise RuntimeError("RG2 state feature contract failure")
    # The state source is keyed by sample key and contains fold copies.  A full
    # refit needs one canonical row per decision key, but only if those copies
    # agree exactly; silently taking an arbitrary fold would be unsafe.
    state_disagreement = state.groupby(["trade_date", "stock_code"], sort=False)[state_features].nunique(dropna=False) > 1
    if state_disagreement.any().any():
        raise RuntimeError("RG2 state differs across repeated decision keys")
    state = state.sort_values("sample_key_sha256", kind="mergesort").drop_duplicates(["trade_date", "stock_code"], keep="first")
    joined = canonical[["trade_date", "stock_code", "market_volatility_4w"]].merge(labels, on=["trade_date", "stock_code"], how="left", validate="one_to_one").merge(rg3_frame[["trade_date", "stock_code", *rg3.DAILY_TECHNICAL_FEATURES]], on=["trade_date", "stock_code"], how="left", validate="one_to_one").merge(state[["trade_date", "stock_code", *state_features]], on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    location = [*rg3.DAILY_TECHNICAL_FEATURES, *state_features]; scale = ["realized_volatility_20d", "downside_volatility_60d", "market_volatility_4w"]
    if joined[[*location, "target_valid"]].isna().any().any() or np.isinf(joined[scale].apply(pd.to_numeric, errors="coerce")).any().any():
        raise RuntimeError("full C1 input join/scale contract failure")
    fit = joined[joined.target_valid.astype(bool)].copy()
    calibration = calibrate(fit, location, scale, hetero, ordinal)
    model = hetero.fit_heteroscedastic_proportional_odds(fit[location].to_numpy(float), fit[scale].to_numpy(float), fit.ordinal_target.astype(int).to_numpy(), location_l2=0.001, scale_l2=0.01, max_iter=200)
    prior = np.bincount(fit.ordinal_target.astype(int), minlength=3).astype(float); prior /= prior.sum()
    output.mkdir(parents=True)
    model_path = output / "C1_HETEROSCEDASTIC_ORDINAL_FULL_DEVELOPMENT.npz"
    np.savez_compressed(model_path, beta=model.beta, gamma=model.gamma, thresholds=np.asarray(model.thresholds), location_median=model.location_scaler.median, location_scale=model.location_scaler.scale, scale_median=model.scale_scaler.median, scale_scale=model.scale_scaler.scale, class_prior=prior)
    artifact = {"artifact_id": "WP22_C1_FULL_DEVELOPMENT_INFERENCE_ARTIFACT_V1", "status": "NON_PRODUCTION_LABEL_FREE_SHADOW_ARTIFACT", "candidate_id": "REV8_C1_RG2_STATE_AUGMENTED_HETEROSKEDASTIC_ORDINAL", "target_id": "T2_MARKET_RELATIVE_FIXED", "model_file": model_path.name, "model_sha256": sha256(model_path), "location_features": location, "scale_features": scale, "calibration": {key: calibration[key] for key in ("temperature", "prior_shrinkage", "inner_oof_rows", "inner_split_count")}, "training_rows": int(len(fit)), "development_origins": int(fit.trade_date.nunique()), "input_sha256": {"samples": sha256(samples_path), "weekly": sha256(weekly_path), "rg3": sha256(rg3_path), "rg2": sha256(rg2_path), "protocol": sha256(protocol_path), "heteroscedastic_source": sha256(hetero_path)}, "fresh_labels_read": False, "wp10_outputs_read": False, "metrics_read": False, "production_replacement_allowed": False, "gpu_used": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    artifact_path = output / "WP22_C1_FULL_DEVELOPMENT_ARTIFACT.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "training_rows": artifact["training_rows"], "model_sha256": artifact["model_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


