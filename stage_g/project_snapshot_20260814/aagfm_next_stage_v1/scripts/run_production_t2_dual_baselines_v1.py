from __future__ import annotations

"""Seal six-fold development predictions for the fixed production-T2 baselines.

This runner is intentionally metric-blind: it writes no validation target, no
validation metric, and no model-selection decision.  It rejects FRESH/SCREENING/
FINAL/sealed paths before opening any input and uses CPU-only proportional odds.
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


EXPECTED_SHA256 = {
    "samples": "60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6",
    "weekly": "4633c51055154309a9af766ea51c75f545783c82f4046261cc211f6a8449815f",
    "features": "04f6b11b7296aa1d92bdc0a97d652565672ac90b1a01601b67f9a629989a9525",
    "split_contract": "d469af80bb77e6f521e293799744d2f972010f67c0416b7d58b5124bd6273b6b",
}
EXPECTED_FOLDS = tuple(f"REV2_RO_{index:02d}" for index in range(1, 7))
BANNED_PATH_TOKENS = ("fresh", "screening", "final", "sealed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_banned_path(path: Path, label: str) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in BANNED_PATH_TOKENS):
        raise RuntimeError(f"{label} resolves to prohibited path token")


def load_module(name: str, path: Path, source_root: Path):
    sys.dont_write_bytecode = True
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules during
    # module execution; register this read-only source module before exec.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_equal_across_duplicates(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        counts = frame.groupby(["trade_date", "stock_code"], sort=False)[column].nunique(dropna=False)
        if int((counts > 1).sum()) != 0:
            raise RuntimeError(f"non-canonical sample value for {column}")


def build_t2_labels(samples: pd.DataFrame, target_module) -> pd.DataFrame:
    canonical_columns = ["trade_date", "stock_code", "target_return_h4", "target_valid", "realized_volatility_8w"]
    assert_equal_across_duplicates(samples, canonical_columns[2:])
    canonical = samples[canonical_columns].drop_duplicates(["trade_date", "stock_code"], keep="first").copy()
    variants = target_module.build_target_variants(canonical)
    labels = variants[["trade_date", "stock_code", "T2_return", "T2_threshold", "T2_valid", "T2_label"]].rename(
        columns={"T2_return": "target_return_h4", "T2_threshold": "target_threshold", "T2_valid": "target_valid", "T2_label": "ordinal_target"}
    )
    if labels.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("T2 label construction produced duplicate keys")
    if set(pd.to_numeric(labels["target_threshold"], errors="coerce").dropna().unique()) != {0.01}:
        raise RuntimeError("T2 threshold is not the frozen fixed 1 percent")
    return labels


def input_audit(samples: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    rows = len(samples)
    if samples["sample_key_sha256"].duplicated().any():
        raise RuntimeError("duplicate sample keys")
    if tuple(sorted(samples["fold_id"].unique())) != EXPECTED_FOLDS:
        raise RuntimeError("unexpected fold identifiers")
    if set(samples["split_role"].unique()) != {"TRAIN", "VALIDATION"}:
        raise RuntimeError("prohibited or missing split role")
    if features.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("duplicate feature keys")
    if (features["source_trade_date"].notna() & (features["source_trade_date"] > features["trade_date"])).any():
        raise RuntimeError("future RG3 source date")
    feature_columns = [column for column in features.columns if column not in {"trade_date", "stock_code", "source_trade_date"}]
    if len(feature_columns) != 14 or features[feature_columns].isna().any().any():
        raise RuntimeError("RG3 technical feature completeness failure")
    joined = samples[["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256"]].merge(
        labels, on=["trade_date", "stock_code"], how="left", validate="many_to_one", indicator="target_join"
    ).merge(
        features[["trade_date", "stock_code", "source_trade_date", *feature_columns]],
        on=["trade_date", "stock_code"], how="left", validate="many_to_one", indicator="feature_join"
    )
    if (joined["target_join"] != "both").any() or (joined["feature_join"] != "both").any():
        raise RuntimeError("missing target or feature after key join")
    if joined[feature_columns].isna().any().any():
        raise RuntimeError("missing RG3 feature after join")
    positions = {
        pd.Timestamp(row.trade_date).normalize(): int(row.position)
        for row in weekly[["trade_date"]].drop_duplicates().sort_values("trade_date").reset_index(names="position").itertuples(index=False)
    }
    split_checks = {}
    for fold in EXPECTED_FOLDS:
        train_dates = pd.DatetimeIndex(sorted(samples.loc[(samples.fold_id == fold) & (samples.split_role == "TRAIN"), "trade_date"].unique()))
        validation_dates = pd.DatetimeIndex(sorted(samples.loc[(samples.fold_id == fold) & (samples.split_role == "VALIDATION"), "trade_date"].unique()))
        train_end = pd.Timestamp(train_dates.max()).normalize()
        validation_start = pd.Timestamp(validation_dates.min()).normalize()
        if train_end not in positions or validation_start not in positions or positions[train_end] + 7 >= positions[validation_start]:
            raise RuntimeError(f"purge/embargo split violation: {fold}")
        split_checks[fold] = {
            "train_origins": int(len(train_dates)), "validation_origins": int(len(validation_dates)),
            "train_end": train_end.date().isoformat(), "validation_start": validation_start.date().isoformat(),
            "origin_position_gap": int(positions[validation_start] - positions[train_end]),
        }
    return {
        "samples_rows": int(rows), "unique_sample_keys": True, "canonical_target_keys": int(len(labels)),
        "features_rows": int(len(features)), "feature_columns": feature_columns,
        "target_feature_join_rows": int(len(joined)), "target_feature_join_complete": True,
        "future_source_trade_date_rows": 0, "split_checks": split_checks,
    }, joined, feature_columns


def probability_frame(base: pd.DataFrame, baseline_id: str, probability: np.ndarray) -> pd.DataFrame:
    if probability.shape != (len(base), 3) or not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("invalid probability output")
    output = base[["fold_id", "trade_date", "stock_code", "sample_key_sha256"]].copy()
    output["baseline_id"] = baseline_id
    output["prob_down"] = probability[:, 0]
    output["prob_neutral"] = probability[:, 1]
    output["prob_up"] = probability[:, 2]
    output["predicted_ordinal"] = np.argmax(probability, axis=1).astype(np.int8)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--feature-input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    feature_input = args.feature_input.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    for label, path in (("source_root", source_root), ("feature_input", feature_input), ("output", output)):
        reject_banned_path(path, label)
    output.mkdir(parents=True)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    samples_path = source_root / "data/rg1_4_materialized/samples.csv.gz"
    weekly_path = source_root / "data/rg1_4_materialized/weekly_panel.csv.gz"
    contract_path = source_root / "governance/rev7_1_freeze/SPLIT_PURGE_EMBARGO_AND_SAMPLE_KEY_CONTRACT.json"
    for label, path, expected in (("samples", samples_path, EXPECTED_SHA256["samples"]), ("weekly", weekly_path, EXPECTED_SHA256["weekly"]), ("features", feature_input, EXPECTED_SHA256["features"]), ("split_contract", contract_path, EXPECTED_SHA256["split_contract"])):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen {label} input identity mismatch")
    target_module = load_module("wp08_rev8_targets", source_root / "src/rev8_targets.py", source_root)
    ordinal_module = load_module("wp08_ordinal", source_root / "src/rg2_calibrated_ordinal.py", source_root)
    rg3_module = load_module("wp08_rg3", source_root / "src/rg3_features.py", source_root)
    usecols = ["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256", "target_return_h4", "target_valid", "realized_volatility_8w"]
    samples = pd.read_csv(samples_path, usecols=usecols, dtype={"fold_id": str, "split_role": str, "stock_code": str})
    samples["trade_date"] = pd.to_datetime(samples["trade_date"], errors="raise").dt.normalize()
    features = pd.read_csv(feature_input, usecols=["trade_date", "stock_code", "source_trade_date", *rg3_module.DAILY_TECHNICAL_FEATURES], dtype={"stock_code": str})
    features["trade_date"] = pd.to_datetime(features["trade_date"], errors="raise").dt.normalize()
    features["source_trade_date"] = pd.to_datetime(features["source_trade_date"], errors="raise").dt.normalize()
    weekly = pd.read_csv(weekly_path, usecols=["trade_date"])
    weekly["trade_date"] = pd.to_datetime(weekly["trade_date"], errors="raise").dt.normalize()
    labels = build_t2_labels(samples, target_module)
    preconsumption, joined, feature_columns = input_audit(samples, labels, features, weekly)
    input_hashes = {"samples": sha256(samples_path), "weekly": sha256(weekly_path), "features": sha256(feature_input), "split_contract": sha256(contract_path), "rev8_targets_source": sha256(source_root / "src/rev8_targets.py"), "ordinal_source": sha256(source_root / "src/rg2_calibrated_ordinal.py"), "rg3_source": sha256(source_root / "src/rg3_features.py")}
    stable_json(output / "INPUT_AND_PATH_DENYLIST_AUDIT.json", {"status": "PASS_READY_FOR_BASELINE_FIT", "input_hashes": input_hashes, "preconsumption": preconsumption, "fresh_payloads_opened": False, "screening_read": False, "final_read": False, "model_trained": False, "gpu_used": False})
    prediction_dir = output / "predictions_sealed"
    prediction_dir.mkdir()
    model_dir = output / "run_receipts"
    model_dir.mkdir()
    all_prediction_paths = []
    fold_receipts = []
    for fold in EXPECTED_FOLDS:
        train = joined[(joined.fold_id == fold) & (joined.split_role == "TRAIN")].copy()
        validation = joined[(joined.fold_id == fold) & (joined.split_role == "VALIDATION")].copy()
        train_valid = train[train.target_valid.astype(bool)].copy()
        if train_valid.empty or validation.empty:
            raise RuntimeError(f"empty train or validation panel: {fold}")
        y = train_valid.ordinal_target.astype(int).to_numpy()
        prior = (np.bincount(y, minlength=3).astype(np.float64) / len(y)).reshape(1, 3)
        ordinal = ordinal_module.fit_proportional_odds(train_valid[feature_columns].to_numpy(float), y, l2=0.001, max_iter=200)
        probability_sets = {
            "NAIVE_PRIOR": np.repeat(prior, len(validation), axis=0),
            "NAIVE_NEUTRAL": np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (len(validation), 1)),
            "INCUMBENT_ORDINAL_T2_DEV": ordinal.predict_proba(validation[feature_columns].to_numpy(float)),
        }
        fold_paths = []
        for baseline_id, probability in probability_sets.items():
            pred = probability_frame(validation, baseline_id, probability).sort_values("sample_key_sha256", kind="mergesort")
            path = prediction_dir / f"{fold}_{baseline_id}.parquet"
            pred.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
            fold_paths.append({"baseline_id": baseline_id, "path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha256(path), "rows": int(len(pred))})
            all_prediction_paths.append(path)
        receipt = {"fold_id": fold, "train_rows": int(len(train)), "train_valid_rows": int(len(train_valid)), "validation_rows": int(len(validation)), "class_prior": prior.reshape(-1).tolist(), "feature_count": len(feature_columns), "ordinal_l2": 0.001, "ordinal_max_iter": 200, "calibration": "none; fixed baseline is metric-blind", "prediction_files": fold_paths, "validation_targets_written": False, "validation_metrics_read": False}
        stable_json(model_dir / f"{fold}.json", receipt)
        fold_receipts.append(receipt)
    sealed = {str(path.relative_to(output)).replace("\\", "/"): sha256(path) for path in sorted(all_prediction_paths)}
    stable_json(output / "PREDICTION_SEAL_MANIFEST.json", {"node_id": "WP08_T2_DUAL_BASELINE_PREDICTION_SEAL", "status": "SEALED_PENDING_INDEPENDENT_METRIC_READ", "input_hashes": input_hashes, "prediction_sha256": sealed, "folds": list(EXPECTED_FOLDS), "baseline_ids": ["NAIVE_PRIOR", "NAIVE_NEUTRAL", "INCUMBENT_ORDINAL_T2_DEV"], "fresh_payloads_opened": False, "validation_targets_written": False, "validation_metrics_read": False, "model_trained": True, "gpu_used": False, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    stable_json(output / "EXECUTION_RECEIPT.json", {"node_id": "WP08_T2_DUAL_BASELINES_CPU_V1", "status": "PASS_PREDICTIONS_SEALED_PENDING_INDEPENDENT_METRIC_READ", "fold_receipts": fold_receipts, "fresh_payloads_opened": False, "screening_read": False, "final_read": False, "model_trained": True, "gpu_used": False, "production_assets_modified": False, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "PASS_PREDICTIONS_SEALED_PENDING_INDEPENDENT_METRIC_READ", "output_root": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


