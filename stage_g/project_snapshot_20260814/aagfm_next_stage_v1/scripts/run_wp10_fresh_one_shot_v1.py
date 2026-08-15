from __future__ import annotations

"""One-shot WP10 FRESH prediction sealing for the frozen C0 candidate.

FRESH labels are deliberately excluded from all payload reads in this runner.
Only an independent evaluator may read those columns after prediction hashes seal.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FRESH = {
    "fresh1": "7706f1dbeebc1e065fbf55266443393a87adbf54702c4162c7d4c02841b30226",
    "fresh2": "f55914c7527df38c95fc294e29d6d9644f91c1e6e6c26ebe6184eaa39b6b61e4",
    "fresh3": "dc4b7f60157b35054a845c2223f6e50a680801a9fbbbf1bd1f371ccb4bedb688",
}
SAMPLES_SHA = "60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6"
WEEKLY_SHA = "4633c51055154309a9af766ea51c75f545783c82f4046261cc211f6a8449815f"
INCUMBENT_ID = "RG_OBGNET_CONFIRMED_SAFE_V1_1"
INCUMBENT_SHA = "d8e4316d0fab70d3785b775c695a1f3a31225edf441a3603e3830f7351c4e2e8"
MODEL_IDS = (
    "REV8_C0_TARGET_ADAPTED_HETEROSKEDASTIC_ORDINAL",
    "NAIVE_PRIOR",
    "NAIVE_NEUTRAL",
    INCUMBENT_ID,
)


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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def incumbent_probability(payload: dict, values: np.ndarray) -> np.ndarray:
    """Read-only reproduction of the active local incumbent probability path."""
    dense = payload["dense_model"]
    median = np.asarray(dense["scaler_median"], dtype=float)
    scale = np.asarray(dense["scaler_scale"], dtype=float)
    beta = np.asarray(dense["beta"], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(beta) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("incumbent feature/scaler contract failure")
    scaled = (values - median) / scale
    scaled[~np.isfinite(scaled)] = 0.0
    scaled = np.clip(scaled, -float(dense["scaler_clip"]), float(dense["scaler_clip"]))
    eta = scaled @ beta
    first, second = (float(item) for item in dense["thresholds"])
    c0 = sigmoid(first - eta); c1 = sigmoid(second - eta)
    raw = np.column_stack([c0, c1 - c0, 1.0 - c1])
    raw = np.clip(raw, 1e-12, None); raw /= raw.sum(axis=1, keepdims=True)
    calibration = payload["calibration"]
    temperature = float(calibration["temperature"]); shrinkage = float(calibration["prior_shrinkage"])
    softened = np.exp(np.log(raw) / temperature - (np.log(raw) / temperature).max(axis=1, keepdims=True))
    softened /= softened.sum(axis=1, keepdims=True)
    prior = np.asarray(calibration["class_prior"], dtype=float).reshape(1, 3)
    probability = (1.0 - shrinkage) * softened + shrinkage * prior
    return probability / probability.sum(axis=1, keepdims=True)


def prediction_frame(base: pd.DataFrame, model_id: str, probability: np.ndarray) -> pd.DataFrame:
    if probability.shape != (len(base), 3) or not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError(f"invalid probability output: {model_id}")
    prediction = base[["trade_date", "stock_code", "sample_key_sha256"]].copy()
    prediction["model_id"] = model_id
    prediction["prob_down"] = probability[:, 0]; prediction["prob_neutral"] = probability[:, 1]; prediction["prob_up"] = probability[:, 2]
    prediction["predicted_ordinal"] = np.argmax(probability, axis=1).astype(np.int8)
    return prediction


def assert_canonical(frame: pd.DataFrame, value_columns: list[str]) -> None:
    for column in value_columns:
        values = frame.groupby(["trade_date", "stock_code"], sort=False)[column].nunique(dropna=False)
        if (values > 1).any():
            raise RuntimeError(f"noncanonical development source: {column}")


def select_calibration(dev: pd.DataFrame, xcols: list[str], zcols: list[str], hetero, ordinal) -> dict:
    splits = ordinal.expanding_time_calibration_splits(dev.trade_date.to_numpy(), folds=3, embargo_weeks=11, minimum_fit_weeks=26)
    probabilities, targets, priors = [], [], []
    for fit_mask, cal_mask in splits:
        fit, cal = dev.loc[fit_mask], dev.loc[cal_mask]
        model = hetero.fit_heteroscedastic_proportional_odds(
            fit[xcols].to_numpy(float), fit[zcols].to_numpy(float), fit.ordinal_target.astype(int),
            location_l2=0.001, scale_l2=0.01, max_iter=200,
        )
        probabilities.append(model.predict_proba(cal[xcols].to_numpy(float), cal[zcols].to_numpy(float)))
        targets.append(cal.ordinal_target.astype(int).to_numpy())
        prior = np.bincount(fit.ordinal_target.astype(int), minlength=3).astype(float); prior /= prior.sum()
        priors.append(np.repeat(prior.reshape(1, 3), len(cal), axis=0))
    probability = np.vstack(probabilities); target = np.concatenate(targets); prior = np.vstack(priors)
    candidates = []
    for temperature in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        for shrinkage in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5):
            calibrated = ordinal.temperature_and_prior_calibrate(probability, prior, temperature, shrinkage)
            brier = float(np.mean(np.sum((calibrated - np.eye(3)[target]) ** 2, axis=1)))
            prediction = np.argmax(calibrated, axis=1); matrix = np.zeros((3, 3), dtype=float); np.add.at(matrix, (target, prediction), 1)
            total = matrix.sum(); target_sum = matrix.sum(1); prediction_sum = matrix.sum(0)
            denom = np.sqrt((total**2 - np.dot(target_sum, target_sum)) * (total**2 - np.dot(prediction_sum, prediction_sum)))
            mcc = float((np.trace(matrix) * total - np.dot(target_sum, prediction_sum)) / denom) if denom else 0.0
            candidates.append({"temperature": temperature, "prior_shrinkage": shrinkage, "brier": brier, "mcc": mcc})
    minimum = min(item["brier"] for item in candidates)
    eligible = [item for item in candidates if item["brier"] <= minimum + 0.005]
    chosen = min(eligible, key=lambda item: (-item["mcc"], item["temperature"], item["prior_shrinkage"]))
    return {"temperature": chosen["temperature"], "prior_shrinkage": chosen["prior_shrinkage"], "inner_oof_rows": int(len(target)), "inner_split_count": len(splits)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--fresh-root", required=True, type=Path)
    parser.add_argument("--feature-input", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--remediation", required=True, type=Path)
    parser.add_argument("--incumbent-model", required=True, type=Path)
    parser.add_argument("--market-state-input", required=True, type=Path)
    parser.add_argument("--market-state-freeze", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source, fresh_root, feature_path, protocol_path, remediation_path, incumbent_path, market_state_path, market_freeze_path, output = (args.source_root.resolve(), args.fresh_root.resolve(), args.feature_input.resolve(), args.protocol.resolve(), args.remediation.resolve(), args.incumbent_model.resolve(), args.market_state_input.resolve(), args.market_state_freeze.resolve(), args.output_root.resolve())
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    if any(token in str(path).lower() for path in (source, fresh_root, feature_path, protocol_path, remediation_path, incumbent_path, market_state_path, market_freeze_path, output) for token in ("screening", "final", "sealed_holdout")):
        raise RuntimeError("prohibited path token")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); remediation = json.loads(remediation_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_CANDIDATE_TRAINING_OR_CANDIDATE_METRIC_READ" or not remediation["status"].startswith("FROZEN_AFTER_PRECONSUMPTION"):
        raise RuntimeError("C0 protocol/remediation not frozen")
    samples_path = source / "data/rg1_4_materialized/samples.csv.gz"
    weekly_path = source / "data/rg1_4_materialized/weekly_panel.csv.gz"
    if sha256(samples_path) != SAMPLES_SHA:
        raise RuntimeError("development sample hash mismatch")
    if sha256(weekly_path) != WEEKLY_SHA:
        raise RuntimeError("development weekly panel hash mismatch")
    fresh_paths = {"fresh1": fresh_root / "fresh1_confirmation.csv.gz", "fresh2": fresh_root / "fresh2_confirmation.csv.gz", "fresh3": fresh_root / "fresh3_incumbent_confirmation.csv.gz"}
    for name, path in fresh_paths.items():
        if not path.is_file() or sha256(path) != FRESH[name]:
            raise RuntimeError(f"FRESH hash mismatch: {name}")
    if not incumbent_path.is_file() or sha256(incumbent_path) != INCUMBENT_SHA:
        raise RuntimeError("active incumbent identity mismatch")
    if not market_state_path.is_file() or not market_freeze_path.is_file():
        raise RuntimeError("missing separately frozen FRESH market-state supplement")
    market_freeze = json.loads(market_freeze_path.read_text(encoding="utf-8"))
    if market_freeze.get("node_id") != "WP10_FRESH_MARKET_VOLATILITY_SUPPLEMENT_FREEZE_V1" or market_freeze.get("status") != "FROZEN_BEFORE_WP10_FRESH_PREDICTION_SEAL":
        raise RuntimeError("FRESH market-state supplement freeze is not approved")
    if market_freeze.get("market_state_sha256") != sha256(market_state_path) or market_freeze.get("development_weekly_panel_sha256") != WEEKLY_SHA:
        raise RuntimeError("FRESH market-state supplement identity mismatch")
    output.mkdir(parents=True)
    target = load_module("wp10_target", source / "src/rev8_targets.py", source)
    ordinal = load_module("wp10_ordinal", source / "src/rg2_calibrated_ordinal.py", source)
    rg3 = load_module("wp10_rg3", source / "src/rg3_features.py", source)
    hetero = load_module("wp10_hetero", Path(__file__).resolve().parents[1] / "src/heteroscedastic_ordinal_t2_v1.py", Path(__file__).resolve().parents[1])
    sample_columns = ["trade_date", "stock_code", "sample_key_sha256", "target_return_h4", "target_valid", "realized_volatility_8w", "market_volatility_4w"]
    samples = pd.read_csv(samples_path, usecols=sample_columns, dtype={"stock_code": str}); samples.trade_date = pd.to_datetime(samples.trade_date, errors="raise").dt.normalize()
    assert_canonical(samples, ["target_return_h4", "target_valid", "realized_volatility_8w", "market_volatility_4w"])
    canonical = samples.drop_duplicates(["trade_date", "stock_code"], keep="first")
    weekly = pd.read_csv(weekly_path, usecols=["trade_date", "stock_code", "market_volatility_4w"], dtype={"stock_code": str})
    weekly.trade_date = pd.to_datetime(weekly.trade_date, errors="raise").dt.normalize()
    assert_canonical(weekly, ["market_volatility_4w"])
    development_market_by_date = weekly[["trade_date", "market_volatility_4w"]].drop_duplicates("trade_date", keep="first")
    if len(development_market_by_date) != weekly.trade_date.nunique():
        raise RuntimeError("weekly panel has inconsistent market-volatility value within date")
    development_market = canonical[["trade_date", "stock_code", "market_volatility_4w"]].merge(development_market_by_date, on="trade_date", how="left", suffixes=("_samples", "_weekly"), validate="many_to_one")
    match = np.isclose(development_market.market_volatility_4w_samples.to_numpy(float), development_market.market_volatility_4w_weekly.to_numpy(float), equal_nan=True)
    if not bool(np.all(match)):
        raise RuntimeError("weekly market-volatility source does not reproduce development C0 scale feature")
    market_state = pd.read_csv(market_state_path, usecols=["trade_date", "market_volatility_4w", "source_trade_date"])
    market_state.trade_date = pd.to_datetime(market_state.trade_date, errors="raise").dt.normalize()
    market_state.source_trade_date = pd.to_datetime(market_state.source_trade_date, errors="raise").dt.normalize()
    if market_state.trade_date.duplicated().any() or (market_state.source_trade_date > market_state.trade_date).any():
        raise RuntimeError("FRESH market-state key/PIT contract failure")
    overlap = development_market_by_date.merge(market_state[["trade_date", "market_volatility_4w"]], on="trade_date", how="inner", suffixes=("_development", "_supplement"))
    if len(overlap) and not bool(np.all(np.isclose(overlap.market_volatility_4w_development.to_numpy(float), overlap.market_volatility_4w_supplement.to_numpy(float), equal_nan=True))):
        raise RuntimeError("FRESH market-state supplement fails development-period reproduction")
    variant = target.build_target_variants(canonical[["trade_date", "stock_code", "target_return_h4", "target_valid", "realized_volatility_8w"]])
    labels = variant[["trade_date", "stock_code", "T2_valid", "T2_label"]].rename(columns={"T2_valid": "target_valid", "T2_label": "ordinal_target"})
    xcols = list(rg3.DAILY_TECHNICAL_FEATURES); zcols = ["realized_volatility_20d", "downside_volatility_60d", "market_volatility_4w"]
    incumbent_payload = json.loads(incumbent_path.read_text(encoding="utf-8"))
    if incumbent_payload.get("model_id") != INCUMBENT_ID or incumbent_payload.get("target_id") != "T2_MARKET_RELATIVE_FIXED" or list(incumbent_payload.get("features", [])) != xcols:
        raise RuntimeError("active incumbent model contract mismatch")
    features = pd.read_csv(feature_path, usecols=["trade_date", "stock_code", *xcols], dtype={"stock_code": str}); features.trade_date = pd.to_datetime(features.trade_date, errors="raise").dt.normalize()
    dev = canonical[["trade_date", "stock_code", "market_volatility_4w"]].merge(labels, on=["trade_date", "stock_code"], how="left", validate="one_to_one").merge(features, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    dev = dev[dev.target_valid.astype(bool)].copy()
    if len(dev) == 0 or dev[xcols].isna().any().any() or dev.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("development input contract failure")
    choice = select_calibration(dev, xcols, zcols, hetero, ordinal)
    fitted = hetero.fit_heteroscedastic_proportional_odds(dev[xcols].to_numpy(float), dev[zcols].to_numpy(float), dev.ordinal_target.astype(int), location_l2=0.001, scale_l2=0.01, max_iter=200)
    prior = np.bincount(dev.ordinal_target.astype(int), minlength=3).astype(float); prior /= prior.sum()
    development_keys = set(samples.sample_key_sha256)
    prediction_dir = output / "predictions_sealed"; prediction_dir.mkdir(); hashes, preflight = {}, []
    seen_fresh_keys: set[str] = set()
    for name, path in fresh_paths.items():
        # Header inspection is schema-only; payload read explicitly excludes every label column.
        header = pd.read_csv(path, nrows=0)
        required_schema = {"trade_date", "stock_code", "sample_key_sha256", "target_valid", "ordinal_target", "target_return_h4", *xcols}
        missing = sorted(required_schema - set(header.columns))
        if missing:
            raise RuntimeError(f"FRESH schema missing {name}: {missing}")
        read_columns = ["trade_date", "stock_code", "sample_key_sha256", *xcols]
        fresh = pd.read_csv(path, usecols=read_columns, dtype={"stock_code": str})
        fresh.trade_date = pd.to_datetime(fresh.trade_date, errors="raise").dt.normalize()
        fresh = fresh.merge(market_state[["trade_date", "market_volatility_4w"]], on="trade_date", how="left", validate="many_to_one", indicator="market_feature_join")
        if (fresh.market_feature_join != "both").any():
            raise RuntimeError(f"FRESH market-volatility source coverage failure: {name}")
        fresh = fresh.drop(columns="market_feature_join")
        fresh_keys = set(fresh.sample_key_sha256)
        overlap_development = fresh_keys & development_keys
        overlap_fresh = fresh_keys & seen_fresh_keys
        if fresh.sample_key_sha256.duplicated().any() or overlap_development or overlap_fresh:
            raise RuntimeError(f"FRESH key isolation failure: {name}")
        seen_fresh_keys.update(fresh_keys)
        c0_probability = ordinal.temperature_and_prior_calibrate(fitted.predict_proba(fresh[xcols].to_numpy(float), fresh[zcols].to_numpy(float)), prior, choice["temperature"], choice["prior_shrinkage"])
        probabilities = {
            protocol["candidate"]["id"]: c0_probability,
            "NAIVE_PRIOR": np.repeat(prior.reshape(1, 3), len(fresh), axis=0),
            "NAIVE_NEUTRAL": np.tile(np.array([[0.0, 1.0, 0.0]], dtype=float), (len(fresh), 1)),
            INCUMBENT_ID: incumbent_probability(incumbent_payload, fresh[xcols].to_numpy(float)),
        }
        files = []
        for model_id in MODEL_IDS:
            prediction = prediction_frame(fresh, model_id, probabilities[model_id]).sort_values("sample_key_sha256", kind="mergesort")
            prediction_path = prediction_dir / f"{name}_{model_id}.parquet"
            prediction.to_parquet(prediction_path, index=False, engine="pyarrow", compression="zstd")
            relative = str(prediction_path.relative_to(output)).replace("\\", "/"); hashes[relative] = sha256(prediction_path)
            files.append({"model_id": model_id, "path": relative, "sha256": hashes[relative], "rows": int(len(prediction))})
        preflight.append({"name": name, "rows": int(len(fresh)), "schema_pass": True, "key_unique": True, "development_key_overlap": 0, "other_fresh_key_overlap": 0, "prediction_files": files})
    input_hashes = {"development_samples": sha256(samples_path), "development_weekly_panel": sha256(weekly_path), "rg3_features": sha256(feature_path), "protocol": sha256(protocol_path), "remediation": sha256(remediation_path), "incumbent_model": sha256(incumbent_path), "market_state_input": sha256(market_state_path), "market_state_freeze": sha256(market_freeze_path), **{name: sha256(path) for name, path in fresh_paths.items()}}
    write_json(output / "FRESH_PRECONSUMPTION_AND_SEAL_AUDIT.json", {"status": "PASS_FRESH_PREDICTIONS_SEALED", "input_hashes": input_hashes, "preflight": preflight, "calibration": choice, "model_ids": list(MODEL_IDS), "incumbent_selection_status": "INCUMBENT_PRODUCTION_IN_SAMPLE_REFERENCE_NOT_VALID_FOR_SELECTION", "fresh_payloads_opened": True, "fresh_labels_read": False, "fresh_metrics_read": False, "model_trained": True, "gpu_used": False})
    write_json(output / "PREDICTION_SEAL_MANIFEST.json", {"node_id": "WP10_C0_AND_CONTROLS_FRESH_PREDICTION_SEAL", "status": "SEALED_PENDING_INDEPENDENT_FRESH_METRIC_READ", "candidate_id": protocol["candidate"]["id"], "model_ids": list(MODEL_IDS), "incumbent_selection_status": "INCUMBENT_PRODUCTION_IN_SAMPLE_REFERENCE_NOT_VALID_FOR_SELECTION", "input_hashes": input_hashes, "prediction_sha256": hashes, "fresh_labels_read": False, "fresh_metrics_read": False, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "PASS_FRESH_PREDICTIONS_SEALED", "output_root": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


