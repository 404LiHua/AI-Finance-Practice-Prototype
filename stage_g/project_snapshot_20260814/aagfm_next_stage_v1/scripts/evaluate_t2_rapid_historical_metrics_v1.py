from __future__ import annotations

"""CPU-only research dashboard for sealed, historical T2 rolling predictions.

It deliberately accepts only development predictions and source samples.  It is
not an evaluator for FRESH, SCREENING, FINAL, shadow, or production promotion.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FOLDS = tuple(f"REV2_RO_{number:02d}" for number in range(1, 7))
BASELINES = ("NAIVE_PRIOR", "NAIVE_NEUTRAL", "INCUMBENT_ORDINAL_T2_DEV")
SAMPLES_SHA = "60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6"
EPSILON = 1e-15


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


def prohibited_path(path: Path) -> bool:
    return any(token in str(path).lower() for token in ("fresh", "screening", "final", "sealed_holdout", "shadow"))


def confusion(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (target.astype(int), predicted.astype(int)), 1)
    return matrix


def classification_metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = confusion(target, predicted)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    precision, recall, f1 = [], [], []
    for label in range(3):
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        p = float(tp / (tp + fp)) if tp + fp else 0.0
        r = float(tp / (tp + fn)) if tp + fn else 0.0
        precision.append(p); recall.append(r); f1.append(float(2 * p * r / (p + r)) if p + r else 0.0)
    true_sum = matrix.sum(axis=1).astype(float)
    predicted_sum = matrix.sum(axis=0).astype(float)
    denominator = math.sqrt(float((total**2 - np.dot(true_sum, true_sum)) * (total**2 - np.dot(predicted_sum, predicted_sum))))
    mcc = float((np.trace(matrix) * total - np.dot(true_sum, predicted_sum)) / denominator) if denominator else 0.0
    result = {
        "scored_rows": total,
        "accuracy": float(correct / total) if total else None,
        "balanced_accuracy": float(np.mean(recall)) if total else None,
        "mcc": mcc,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "ordinal_mae": float(np.mean(np.abs(target.astype(float) - predicted.astype(float)))) if total else None,
        "confusion_matrix": matrix.tolist(),
    }
    for label, name in enumerate(("down", "neutral", "up")):
        result[f"precision_{name}"] = precision[label]
        result[f"recall_{name}"] = recall[label]
        result[f"f1_{name}"] = f1[label]
    return result


def probability_metrics(target: np.ndarray, probability: np.ndarray) -> tuple[dict, pd.DataFrame]:
    probability = np.clip(probability, EPSILON, 1.0)
    probability = probability / probability.sum(axis=1, keepdims=True)
    one_hot = np.eye(3)[target.astype(int)]
    confidence = probability.max(axis=1)
    predicted = probability.argmax(axis=1)
    entropy = -np.sum(probability * np.log(probability), axis=1)
    brier = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
    log_loss = float(-np.mean(np.log(probability[np.arange(len(target)), target.astype(int)])))
    bin_rows, ece = [], 0.0
    for bucket in range(10):
        lower, upper = bucket / 10, (bucket + 1) / 10
        mask = (confidence >= lower) & ((confidence < upper) if bucket < 9 else (confidence <= upper))
        count = int(mask.sum())
        if count:
            accuracy = float(np.mean(predicted[mask] == target[mask]))
            mean_confidence = float(np.mean(confidence[mask]))
            ece += count / len(target) * abs(accuracy - mean_confidence)
        else:
            accuracy = None; mean_confidence = None
        bin_rows.append({"bin": bucket, "lower": lower, "upper": upper, "count": count, "mean_confidence": mean_confidence, "empirical_accuracy": accuracy})
    result = {
        "brier": brier,
        "multiclass_log_loss": log_loss,
        "ece_10bin": float(ece),
        "mean_confidence": float(np.mean(confidence)),
        "mean_entropy": float(np.mean(entropy)),
        "predicted_prevalence_down": float(np.mean(predicted == 0)),
        "predicted_prevalence_neutral": float(np.mean(predicted == 1)),
        "predicted_prevalence_up": float(np.mean(predicted == 2)),
        "observed_prevalence_down": float(np.mean(target == 0)),
        "observed_prevalence_neutral": float(np.mean(target == 1)),
        "observed_prevalence_up": float(np.mean(target == 2)),
    }
    return result, pd.DataFrame(bin_rows)


def weekly_metrics(frame: pd.DataFrame, minimum_stocks: int) -> pd.DataFrame:
    rows = []
    valid = frame[frame.target_valid.astype(bool)].copy()
    valid["score"] = valid.prob_up - valid.prob_down
    for trade_date, group in valid.groupby("trade_date", sort=True):
        enough = len(group) >= minimum_stocks
        if enough and group.score.nunique() >= 3 and group.target_return_h4.nunique() >= 3:
            score_rank = group.score.rank(method="average").to_numpy(float)
            target_rank = group.target_return_h4.rank(method="average").to_numpy(float)
            ic = float(np.corrcoef(score_rank, target_rank)[0, 1])
            tenth = max(1, int(math.ceil(len(group) / 10)))
            ordered = group.sort_values(["score", "stock_code"], kind="mergesort")
            top_bottom_spread = float(ordered.tail(tenth).target_return_h4.mean() - ordered.head(tenth).target_return_h4.mean())
        else:
            ic = None; top_bottom_spread = None
        rows.append({"trade_date": pd.Timestamp(trade_date), "valid_rows": int(len(group)), "minimum_universe_pass": bool(enough), "weekly_spearman_ic": ic, "top_bottom_decile_h4_spread": top_bottom_spread})
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, minimum_stocks: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    valid = frame[frame.target_valid.astype(bool)].copy()
    required = ["prob_down", "prob_neutral", "prob_up", "predicted_ordinal"]
    if valid[required].isna().any().any() or not np.isfinite(valid[required].to_numpy(float)).all():
        raise RuntimeError("non-finite prediction values")
    probability = valid[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)
    sum_error = np.abs(probability.sum(axis=1) - 1.0)
    if (sum_error > 1e-8).any() or (probability < -1e-12).any():
        raise RuntimeError("invalid probability simplex")
    target = valid.ordinal_target.to_numpy(int)
    predicted = valid.predicted_ordinal.to_numpy(int)
    result = classification_metrics(target, predicted)
    probability_result, calibration = probability_metrics(target, probability)
    result.update(probability_result)
    result["key_coverage"] = float(len(frame) / len(frame)) if len(frame) else 0.0
    result["valid_target_coverage"] = float(len(valid) / len(frame)) if len(frame) else 0.0
    result["max_probability_sum_error"] = float(sum_error.max()) if len(sum_error) else None
    weekly = weekly_metrics(frame, minimum_stocks)
    qualified = weekly.dropna(subset=["weekly_spearman_ic"])
    values = qualified.weekly_spearman_ic.to_numpy(float)
    result.update({
        "qualified_weeks": int(len(qualified)),
        "mean_weekly_spearman_ic": float(np.mean(values)) if len(values) else None,
        "median_weekly_spearman_ic": float(np.median(values)) if len(values) else None,
        "ic_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "icir_annualized": float(np.mean(values) / np.std(values, ddof=1) * math.sqrt(52)) if len(values) > 1 and np.std(values, ddof=1) > 0 else None,
        "positive_ic_week_rate": float(np.mean(values > 0)) if len(values) else None,
        "mean_top_bottom_decile_h4_spread": float(qualified.top_bottom_decile_h4_spread.mean()) if len(qualified) else None,
        "positive_top_bottom_week_rate": float(np.mean(qualified.top_bottom_decile_h4_spread > 0)) if len(qualified) else None,
        "minimum_universe_compliance": float(weekly.minimum_universe_pass.mean()) if len(weekly) else None,
    })
    return result, weekly, calibration


def build_labels(samples: pd.DataFrame, target_module) -> pd.DataFrame:
    values = ["target_return_h4", "target_valid", "realized_volatility_8w"]
    for value in values:
        if (samples.groupby(["trade_date", "stock_code"], sort=False)[value].nunique(dropna=False) > 1).any():
            raise RuntimeError(f"noncanonical target source: {value}")
    base = samples[["trade_date", "stock_code", *values]].drop_duplicates(["trade_date", "stock_code"], keep="first")
    variants = target_module.build_target_variants(base)
    return variants[["trade_date", "stock_code", "T2_return", "T2_valid", "T2_label"]].rename(columns={"T2_return": "target_return_h4", "T2_valid": "target_valid", "T2_label": "ordinal_target"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source, baseline_run, candidate_run, protocol_path, output = (item.resolve() for item in (args.source_root, args.baseline_run, args.candidate_run, args.protocol, args.output_root))
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    if any(prohibited_path(path) for path in (source, baseline_run, candidate_run, protocol_path, output)):
        raise RuntimeError("prohibited FRESH, screening, final, holdout, or shadow path")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_FOR_RESEARCH_METRIC_MEASUREMENT":
        raise RuntimeError("protocol is not frozen for research measurement")
    baseline_manifest_path = baseline_run / "PREDICTION_SEAL_MANIFEST.json"
    candidate_manifest_path = candidate_run / "PREDICTION_SEAL_MANIFEST.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if baseline_manifest.get("status") != "SEALED_PENDING_INDEPENDENT_METRIC_READ" or candidate_manifest.get("status") != "SEALED_PENDING_INDEPENDENT_METRIC_READ":
        raise RuntimeError("prediction package is not sealed")
    for run, manifest in ((baseline_run, baseline_manifest), (candidate_run, candidate_manifest)):
        for relative, expected in manifest["prediction_sha256"].items():
            if sha256(run / relative) != expected:
                raise RuntimeError(f"prediction hash mismatch: {relative}")
    samples_path = source / "data" / "rg1_4_materialized" / "samples.csv.gz"
    if sha256(samples_path) != SAMPLES_SHA:
        raise RuntimeError("source samples hash mismatch")
    target_module = load_module("wp12_target", source / "src" / "rev8_targets.py", source)
    usecols = ["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256", "target_return_h4", "target_valid", "realized_volatility_8w"]
    samples = pd.read_csv(samples_path, usecols=usecols, dtype={"fold_id": str, "split_role": str, "stock_code": str})
    samples.trade_date = pd.to_datetime(samples.trade_date, errors="raise").dt.normalize()
    labels = build_labels(samples, target_module)
    validation = samples[samples.split_role == "VALIDATION"][["fold_id", "trade_date", "stock_code", "sample_key_sha256"]].merge(labels, on=["trade_date", "stock_code"], how="left", validate="many_to_one")
    frames: dict[str, pd.DataFrame] = {}
    for model_id in BASELINES:
        parts = []
        for fold in FOLDS:
            path = baseline_run / "predictions_sealed" / f"{fold}_{model_id}.parquet"
            prediction = pd.read_parquet(path, engine="pyarrow")
            prediction.trade_date = pd.to_datetime(prediction.trade_date, errors="raise").dt.normalize()
            joined = validation[validation.fold_id == fold].merge(prediction, on=["fold_id", "trade_date", "stock_code", "sample_key_sha256"], how="left", validate="one_to_one", indicator=True)
            if (joined._merge != "both").any() or (joined.baseline_id != model_id).any():
                raise RuntimeError(f"baseline coverage or identity failure: {fold}/{model_id}")
            parts.append(joined.drop(columns="_merge"))
        frames[model_id] = pd.concat(parts, ignore_index=True)
    candidate_id = str(candidate_manifest.get("candidate_id", "REV8_C0_TARGET_ADAPTED_HETEROSKEDASTIC_ORDINAL"))
    candidate_parts = []
    for fold in FOLDS:
        prediction = pd.read_parquet(candidate_run / "predictions_sealed" / f"{fold}_C0.parquet", engine="pyarrow")
        prediction.trade_date = pd.to_datetime(prediction.trade_date, errors="raise").dt.normalize()
        joined = validation[validation.fold_id == fold].merge(prediction, on=["fold_id", "trade_date", "stock_code", "sample_key_sha256"], how="left", validate="one_to_one", indicator=True)
        if (joined._merge != "both").any() or (joined.candidate_id != candidate_id).any():
            raise RuntimeError(f"candidate coverage or identity failure: {fold}")
        candidate_parts.append(joined.drop(columns="_merge"))
    frames[candidate_id] = pd.concat(candidate_parts, ignore_index=True)
    output.mkdir(parents=True)
    minimum_stocks = int(protocol["target"]["minimum_valid_stocks_per_week"])
    global_rows, fold_rows, weekly_rows, calibration_rows = [], [], [], []
    for model_id, frame in frames.items():
        overall, weekly, calibration = aggregate(frame, minimum_stocks)
        overall["model_id"] = model_id; global_rows.append(overall)
        weekly["model_id"] = model_id; weekly_rows.append(weekly)
        calibration["model_id"] = model_id; calibration_rows.append(calibration)
        for fold in FOLDS:
            details, _, _ = aggregate(frame[frame.fold_id == fold], minimum_stocks)
            details.update({"model_id": model_id, "fold_id": fold}); fold_rows.append(details)
    pd.DataFrame(global_rows).sort_values("model_id", kind="mergesort").to_csv(output / "GLOBAL_RESEARCH_METRICS.csv", index=False, encoding="utf-8")
    pd.DataFrame(fold_rows).sort_values(["model_id", "fold_id"], kind="mergesort").to_csv(output / "FOLD_RESEARCH_METRICS.csv", index=False, encoding="utf-8")
    pd.concat(weekly_rows, ignore_index=True).sort_values(["model_id", "trade_date"], kind="mergesort").to_csv(output / "WEEKLY_RESEARCH_METRICS.csv", index=False, encoding="utf-8")
    pd.concat(calibration_rows, ignore_index=True).sort_values(["model_id", "bin"], kind="mergesort").to_csv(output / "CALIBRATION_10BIN.csv", index=False, encoding="utf-8")
    receipt = {
        "node_id": "WP12_RAPID_HISTORICAL_METRICS_V1",
        "status": "COMPLETE_RESEARCH_ONLY_NOT_PROMOTION_EVIDENCE",
        "protocol_sha256": sha256(protocol_path),
        "source_samples_sha256": sha256(samples_path),
        "baseline_manifest_sha256": sha256(baseline_manifest_path),
        "candidate_manifest_sha256": sha256(candidate_manifest_path),
        "models": sorted(frames),
        "fresh_payloads_opened": False,
        "production_replacement_allowed": False,
        "gpu_used": False,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "RESEARCH_METRIC_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output_root": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


