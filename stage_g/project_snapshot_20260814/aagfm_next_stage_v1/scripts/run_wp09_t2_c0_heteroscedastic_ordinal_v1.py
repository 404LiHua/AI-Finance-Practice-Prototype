from __future__ import annotations

"""One-shot, metric-blind execution of the frozen WP09 C0 candidate."""

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


FOLDS = tuple(f"REV2_RO_{i:02d}" for i in range(1, 7))
EXPECTED = {
    "samples": "60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6",
    "weekly": "4633c51055154309a9af766ea51c75f545783c82f4046261cc211f6a8449815f",
    "features": "04f6b11b7296aa1d92bdc0a97d652565672ac90b1a01601b67f9a629989a9525",
    "protocol": "",  # Filled by the executor after reading this immutable local protocol.
}
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
    matrix = np.zeros((3, 3), dtype=float)
    np.add.at(matrix, (np.asarray(target, dtype=int), np.asarray(prediction, dtype=int)), 1)
    total = matrix.sum(); true_sum = matrix.sum(1); pred_sum = matrix.sum(0)
    denominator = np.sqrt((total**2 - np.dot(true_sum, true_sum)) * (total**2 - np.dot(pred_sum, pred_sum)))
    return float((np.trace(matrix) * total - np.dot(true_sum, pred_sum)) / denominator) if denominator else 0.0


def brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.sum((probability - np.eye(3)[np.asarray(target, dtype=int)]) ** 2, axis=1)))


def build_labels(samples: pd.DataFrame, target_module) -> pd.DataFrame:
    fields = ["target_return_h4", "target_valid", "realized_volatility_8w"]
    for field in fields:
        if (samples.groupby(["trade_date", "stock_code"], sort=False)[field].nunique(dropna=False) > 1).any():
            raise RuntimeError(f"noncanonical target field: {field}")
    canonical = samples[["trade_date", "stock_code", *fields]].drop_duplicates(["trade_date", "stock_code"], keep="first")
    variants = target_module.build_target_variants(canonical)
    return variants[["trade_date", "stock_code", "T2_return", "T2_valid", "T2_label"]].rename(columns={"T2_return": "target_return_h4", "T2_valid": "target_valid", "T2_label": "ordinal_target"})


def calibrate(train: pd.DataFrame, location_columns: list[str], scale_columns: list[str], hetero, ordinal) -> dict:
    dates = train.trade_date.to_numpy()
    splits = ordinal.expanding_time_calibration_splits(dates, folds=3, embargo_weeks=11, minimum_fit_weeks=26)
    probability_parts = []; target_parts = []; prior_parts = []
    for fit_mask, cal_mask in splits:
        fit = train.loc[fit_mask]; cal = train.loc[cal_mask]
        if fit.empty or cal.empty:
            raise RuntimeError("empty frozen inner calibration split")
        fitted = hetero.fit_heteroscedastic_proportional_odds(fit[location_columns].to_numpy(float), fit[scale_columns].to_numpy(float), fit.ordinal_target.astype(int).to_numpy(), location_l2=0.001, scale_l2=0.01, max_iter=200)
        probability_parts.append(fitted.predict_proba(cal[location_columns].to_numpy(float), cal[scale_columns].to_numpy(float)))
        target_parts.append(cal.ordinal_target.astype(int).to_numpy())
        prior = np.bincount(fit.ordinal_target.astype(int), minlength=3).astype(float); prior /= prior.sum()
        prior_parts.append(np.repeat(prior.reshape(1, 3), len(cal), axis=0))
    probability = np.vstack(probability_parts); target = np.concatenate(target_parts); prior = np.vstack(prior_parts)
    candidates = []
    for temperature in TEMPERATURES:
        for shrinkage in SHRINKAGES:
            calibrated = ordinal.temperature_and_prior_calibrate(probability, prior, temperature, shrinkage)
            candidates.append({"temperature": temperature, "prior_shrinkage": shrinkage, "brier": brier(target, calibrated), "mcc": mcc(target, np.argmax(calibrated, axis=1))})
    minimum_brier = min(item["brier"] for item in candidates)
    admissible = [item for item in candidates if item["brier"] <= minimum_brier + 0.005]
    chosen = min(admissible, key=lambda item: (-item["mcc"], item["temperature"], item["prior_shrinkage"]))
    return {"temperature": chosen["temperature"], "prior_shrinkage": chosen["prior_shrinkage"], "inner_oof_rows": int(len(target)), "inner_split_count": len(splits)}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--feature-input", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--missingness-remediation", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve(); features_path = args.feature_input.resolve(); protocol_path = args.protocol.resolve(); remediation_path = args.missingness_remediation.resolve(); output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    if any(token in str(path).lower() for path in (source, features_path, protocol_path, remediation_path, output) for token in ("fresh", "screening", "final", "sealed_holdout")):
        raise RuntimeError("prohibited path token")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_TRAINING_OR_CANDIDATE_METRIC_READ":
        raise RuntimeError("candidate protocol is not frozen")
    remediation = json.loads(remediation_path.read_text(encoding="utf-8"))
    if remediation.get("status") != "FROZEN_AFTER_PRECONSUMPTION_FAILURE_BEFORE_ANY_CANDIDATE_FIT_OR_SEALED_PREDICTION":
        raise RuntimeError("scale-feature remediation is not frozen")
    output.mkdir(parents=True)
    os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    samples_path = source / "data/rg1_4_materialized/samples.csv.gz"; weekly_path = source / "data/rg1_4_materialized/weekly_panel.csv.gz"
    for label, path in (("samples", samples_path), ("weekly", weekly_path), ("features", features_path)):
        if not path.is_file() or sha256(path) != EXPECTED[label]:
            raise RuntimeError(f"{label} hash mismatch")
    target = load_module("wp09_rev8_target", source / "src/rev8_targets.py", source)
    rg3 = load_module("wp09_rg3", source / "src/rg3_features.py", source)
    ordinal = load_module("wp09_ordinal", source / "src/rg2_calibrated_ordinal.py", source)
    hetero = load_module("wp09_hetero", Path(__file__).resolve().parents[1] / "src/heteroscedastic_ordinal_t2_v1.py", Path(__file__).resolve().parents[1])
    usecols = ["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256", "target_return_h4", "target_valid", "realized_volatility_8w", "market_volatility_4w"]
    samples = pd.read_csv(samples_path, usecols=usecols, dtype={"fold_id": str, "split_role": str, "stock_code": str})
    samples.trade_date = pd.to_datetime(samples.trade_date, errors="raise").dt.normalize()
    if tuple(sorted(samples.fold_id.unique())) != FOLDS or set(samples.split_role.unique()) != {"TRAIN", "VALIDATION"} or samples.sample_key_sha256.duplicated().any():
        raise RuntimeError("sample split/key contract failure")
    weekly_dates = pd.read_csv(weekly_path, usecols=["trade_date"])
    weekly_dates["trade_date"] = pd.to_datetime(weekly_dates["trade_date"], errors="raise").dt.normalize()
    origin_position = {date: position for position, date in enumerate(sorted(weekly_dates.trade_date.unique()))}
    for fold in FOLDS:
        train_end = samples.loc[(samples.fold_id == fold) & (samples.split_role == "TRAIN"), "trade_date"].max()
        validation_start = samples.loc[(samples.fold_id == fold) & (samples.split_role == "VALIDATION"), "trade_date"].min()
        if origin_position[pd.Timestamp(train_end)] + 7 >= origin_position[pd.Timestamp(validation_start)]:
            raise RuntimeError(f"purge/embargo contract failure: {fold}")
    features = pd.read_csv(features_path, usecols=["trade_date", "stock_code", "source_trade_date", *rg3.DAILY_TECHNICAL_FEATURES], dtype={"stock_code": str})
    features.trade_date = pd.to_datetime(features.trade_date, errors="raise").dt.normalize(); features.source_trade_date = pd.to_datetime(features.source_trade_date, errors="raise").dt.normalize()
    if features.duplicated(["trade_date", "stock_code"]).any() or (features.source_trade_date > features.trade_date).any() or features[rg3.DAILY_TECHNICAL_FEATURES].isna().any().any():
        raise RuntimeError("RG3 feature/PIT contract failure")
    labels = build_labels(samples, target)
    identity_and_scale = samples[["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256", "market_volatility_4w"]]
    joined = identity_and_scale.merge(labels, on=["trade_date", "stock_code"], how="left", validate="many_to_one", indicator="target_join").merge(features[["trade_date", "stock_code", *rg3.DAILY_TECHNICAL_FEATURES]], on=["trade_date", "stock_code"], how="left", validate="many_to_one", indicator="feature_join")
    if (joined.target_join != "both").any() or (joined.feature_join != "both").any() or joined[rg3.DAILY_TECHNICAL_FEATURES].isna().any().any():
        raise RuntimeError("candidate input join completeness failure")
    location_columns = list(rg3.DAILY_TECHNICAL_FEATURES); scale_columns = ["realized_volatility_20d", "downside_volatility_60d", "market_volatility_4w"]
    for column in scale_columns:
        if np.isinf(pd.to_numeric(joined[column], errors="coerce")).any():
            raise RuntimeError(f"infinite scale feature: {column}")
    scale_missing = {column: int(pd.to_numeric(joined[column], errors="coerce").isna().sum()) for column in scale_columns}
    input_hashes = {"samples": sha256(samples_path), "weekly": sha256(weekly_path), "features": sha256(features_path), "protocol": sha256(protocol_path), "missingness_remediation": sha256(remediation_path), "candidate_source": sha256(Path(__file__).resolve().parents[1] / "src/heteroscedastic_ordinal_t2_v1.py")}
    write_json(output / "PRECONSUMPTION_AUDIT.json", {"status": "PASS_READY_FOR_C0_FIT", "input_hashes": input_hashes, "folds": list(FOLDS), "location_columns": location_columns, "scale_columns": scale_columns, "scale_feature_missing_rows": scale_missing, "missingness_policy": remediation["frozen_remediation"], "fresh_payloads_opened": False, "screening_read": False, "final_read": False, "candidate_metrics_read": False, "gpu_used": False})
    prediction_dir = output / "predictions_sealed"; prediction_dir.mkdir(); receipt_dir = output / "run_receipts"; receipt_dir.mkdir()
    prediction_hashes = {}; receipts = []
    for fold in FOLDS:
        train = joined[(joined.fold_id == fold) & (joined.split_role == "TRAIN") & (joined.target_valid.astype(bool))].copy()
        validation = joined[(joined.fold_id == fold) & (joined.split_role == "VALIDATION")].copy()
        choice = calibrate(train, location_columns, scale_columns, hetero, ordinal)
        fitted = hetero.fit_heteroscedastic_proportional_odds(train[location_columns].to_numpy(float), train[scale_columns].to_numpy(float), train.ordinal_target.astype(int).to_numpy(), location_l2=0.001, scale_l2=0.01, max_iter=200)
        prior = np.bincount(train.ordinal_target.astype(int), minlength=3).astype(float); prior /= prior.sum()
        raw = fitted.predict_proba(validation[location_columns].to_numpy(float), validation[scale_columns].to_numpy(float))
        probability = ordinal.temperature_and_prior_calibrate(raw, prior, choice["temperature"], choice["prior_shrinkage"])
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(1), 1.0, atol=1e-12):
            raise RuntimeError(f"probability failure {fold}")
        pred = validation[["fold_id", "trade_date", "stock_code", "sample_key_sha256"]].copy()
        pred["candidate_id"] = protocol["candidate"]["id"]; pred["prob_down"] = probability[:, 0]; pred["prob_neutral"] = probability[:, 1]; pred["prob_up"] = probability[:, 2]; pred["predicted_ordinal"] = np.argmax(probability, axis=1).astype(np.int8)
        pred = pred.sort_values("sample_key_sha256", kind="mergesort")
        path = prediction_dir / f"{fold}_C0.parquet"; pred.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
        prediction_hashes[str(path.relative_to(output)).replace("\\", "/")] = sha256(path)
        receipt = {"fold_id": fold, "train_valid_rows": int(len(train)), "validation_rows": int(len(validation)), "location_feature_count": len(location_columns), "scale_feature_count": len(scale_columns), "scale_feature_missing_rows_train": {column: int(train[column].isna().sum()) for column in scale_columns}, "scale_feature_missing_rows_validation": {column: int(validation[column].isna().sum()) for column in scale_columns}, "calibration": choice, "location_l2": 0.001, "scale_l2": 0.01, "log_scale_bound": 1.0, "validation_targets_written": False, "validation_metrics_read": False, "prediction_sha256": sha256(path)}
        write_json(receipt_dir / f"{fold}.json", receipt); receipts.append(receipt)
    seal = {"node_id": "WP09_C0_PREDICTION_SEAL_V1", "status": "SEALED_PENDING_INDEPENDENT_METRIC_READ", "protocol_sha256": input_hashes["protocol"], "input_hashes": input_hashes, "candidate_id": protocol["candidate"]["id"], "prediction_sha256": prediction_hashes, "folds": list(FOLDS), "validation_targets_written": False, "candidate_metrics_read": False, "fresh_payloads_opened": False, "model_trained": True, "gpu_used": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "PREDICTION_SEAL_MANIFEST.json", seal)
    write_json(output / "EXECUTION_RECEIPT.json", {"node_id": "WP09_C0_EXECUTION_V1", "status": "PASS_C0_PREDICTIONS_SEALED_PENDING_INDEPENDENT_METRIC_READ", "fold_receipts": receipts, "fresh_payloads_opened": False, "screening_read": False, "final_read": False, "model_trained": True, "gpu_used": False, "production_assets_modified": False, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "PASS_C0_PREDICTIONS_SEALED_PENDING_INDEPENDENT_METRIC_READ", "output_root": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


