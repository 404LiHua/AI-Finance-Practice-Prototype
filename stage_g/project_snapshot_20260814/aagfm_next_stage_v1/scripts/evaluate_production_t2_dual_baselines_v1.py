from __future__ import annotations

"""Independent metric reader for a sealed WP08 prediction package."""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SAMPLE_SHA256 = "60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6"
BANNED_PATH_TOKENS = ("fresh", "screening", "final", "sealed_holdout")
BASELINES = ("NAIVE_PRIOR", "NAIVE_NEUTRAL", "INCUMBENT_ORDINAL_T2_DEV")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_banned_path(path: Path, label: str) -> None:
    if any(token in str(path).lower() for token in BANNED_PATH_TOKENS):
        raise RuntimeError(f"{label} resolves to prohibited path token")


def load_module(name: str, path: Path, source_root: Path):
    sys.dont_write_bytecode = True
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load target source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def multiclass_mcc(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.int64)
    p = np.asarray(prediction, dtype=np.int64)
    matrix = np.zeros((3, 3), dtype=np.float64)
    np.add.at(matrix, (y, p), 1.0)
    total = matrix.sum()
    correct = np.trace(matrix)
    true_sum = matrix.sum(axis=1)
    pred_sum = matrix.sum(axis=0)
    numerator = correct * total - np.dot(true_sum, pred_sum)
    denominator = np.sqrt((total**2 - np.dot(pred_sum, pred_sum)) * (total**2 - np.dot(true_sum, true_sum)))
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def metrics(frame: pd.DataFrame) -> dict:
    valid = frame[frame.target_valid.astype(bool)].copy()
    y = valid.ordinal_target.astype(int).to_numpy()
    probability = valid[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)
    predicted = valid.predicted_ordinal.astype(int).to_numpy()
    brier = float(np.mean(np.sum((probability - np.eye(3)[y]) ** 2, axis=1)))
    weekly = []
    valid["score"] = valid.prob_up - valid.prob_down
    for trade_date, group in valid.groupby("trade_date", sort=True):
        if len(group) < 3 or group.score.nunique() < 2 or group.target_return_h4.nunique() < 2:
            continue
        ic = group.score.rank(method="average").corr(group.target_return_h4.rank(method="average"))
        if pd.notna(ic):
            weekly.append(float(ic))
    return {
        "scored_rows": int(len(valid)),
        "mcc": multiclass_mcc(y, predicted),
        "brier": brier,
        "mean_weekly_spearman_ic": float(np.mean(weekly)) if weekly else None,
        "weekly_ic_count": int(len(weekly)),
    }


def build_labels(samples: pd.DataFrame, target_module) -> pd.DataFrame:
    source = samples[["trade_date", "stock_code", "target_return_h4", "target_valid", "realized_volatility_8w"]].copy()
    for column in ("target_return_h4", "target_valid", "realized_volatility_8w"):
        if (source.groupby(["trade_date", "stock_code"], sort=False)[column].nunique(dropna=False) > 1).any():
            raise RuntimeError(f"non-canonical target source: {column}")
    canonical = source.drop_duplicates(["trade_date", "stock_code"], keep="first")
    variants = target_module.build_target_variants(canonical)
    return variants[["trade_date", "stock_code", "T2_return", "T2_valid", "T2_label"]].rename(columns={"T2_return": "target_return_h4", "T2_valid": "target_valid", "T2_label": "ordinal_target"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--sealed-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    sealed_run = args.sealed_run.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite metric output: {output}")
    for label, path in (("source_root", source_root), ("sealed_run", sealed_run), ("output", output)):
        reject_banned_path(path, label)
    manifest_path = sealed_run / "PREDICTION_SEAL_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("sealed prediction manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SEALED_PENDING_INDEPENDENT_METRIC_READ" or manifest.get("validation_metrics_read"):
        raise RuntimeError("predictions are not eligible for independent metric read")
    seal_failures = []
    for relative, expected in manifest["prediction_sha256"].items():
        path = sealed_run / Path(relative)
        if not path.is_file() or sha256(path) != expected:
            seal_failures.append(relative)
    if seal_failures:
        raise RuntimeError(f"sealed prediction identity mismatch: {seal_failures}")
    samples_path = source_root / "data/rg1_4_materialized/samples.csv.gz"
    if sha256(samples_path) != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError("sample source identity mismatch")
    output.mkdir(parents=True)
    target_module = load_module("wp08_independent_rev8_targets", source_root / "src/rev8_targets.py", source_root)
    usecols = ["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256", "target_return_h4", "target_valid", "realized_volatility_8w"]
    samples = pd.read_csv(samples_path, usecols=usecols, dtype={"fold_id": str, "split_role": str, "stock_code": str})
    samples["trade_date"] = pd.to_datetime(samples.trade_date, errors="raise").dt.normalize()
    labels = build_labels(samples, target_module)
    expected_validation = samples[samples.split_role == "VALIDATION"][["fold_id", "trade_date", "stock_code", "sample_key_sha256"]].merge(labels, on=["trade_date", "stock_code"], how="left", validate="many_to_one")
    if expected_validation.target_valid.isna().any():
        raise RuntimeError("target label join incomplete")
    reports = []
    combined = {}
    required_prediction_columns = {"fold_id", "trade_date", "stock_code", "sample_key_sha256", "baseline_id", "prob_down", "prob_neutral", "prob_up", "predicted_ordinal"}
    for baseline in BASELINES:
        parts = []
        for fold in manifest["folds"]:
            path = sealed_run / "predictions_sealed" / f"{fold}_{baseline}.parquet"
            pred = pd.read_parquet(path, engine="pyarrow")
            if set(pred.columns) != required_prediction_columns or len(pred) == 0 or pred.duplicated("sample_key_sha256").any():
                raise RuntimeError(f"prediction schema or keys invalid: {fold}/{baseline}")
            pred["trade_date"] = pd.to_datetime(pred.trade_date, errors="raise").dt.normalize()
            expected = expected_validation[expected_validation.fold_id == fold]
            joined = expected.merge(pred, on=["fold_id", "trade_date", "stock_code", "sample_key_sha256"], how="left", validate="one_to_one", indicator=True)
            if (joined["_merge"] != "both").any():
                raise RuntimeError(f"prediction coverage failure: {fold}/{baseline}")
            if (joined.baseline_id != baseline).any():
                raise RuntimeError(f"baseline identity mismatch: {fold}/{baseline}")
            probability = joined[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)
            if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12):
                raise RuntimeError(f"probability integrity failure: {fold}/{baseline}")
            result = metrics(joined)
            result.update({"baseline_id": baseline, "fold_id": fold})
            reports.append(result)
            parts.append(joined)
        combined[baseline] = pd.concat(parts, ignore_index=True)
    metric_table = pd.DataFrame(reports).sort_values(["baseline_id", "fold_id"], kind="mergesort")
    global_metrics = []
    for baseline, frame in combined.items():
        result = metrics(frame)
        result["baseline_id"] = baseline
        global_metrics.append(result)
    global_table = pd.DataFrame(global_metrics).sort_values("baseline_id", kind="mergesort")
    naive = global_table[global_table.baseline_id.isin(["NAIVE_PRIOR", "NAIVE_NEUTRAL"])].copy()
    maximum = naive.mcc.max()
    naive_winners = naive.loc[np.isclose(naive.mcc, maximum), "baseline_id"].tolist()
    best_naive = naive_winners[0] if len(naive_winners) == 1 else None
    comparisons = []
    ordinal_by_fold = metric_table[metric_table.baseline_id == "INCUMBENT_ORDINAL_T2_DEV"].set_index("fold_id")
    for baseline in ("NAIVE_PRIOR", "NAIVE_NEUTRAL"):
        base = metric_table[metric_table.baseline_id == baseline].set_index("fold_id")
        delta = ordinal_by_fold[["mcc", "brier", "mean_weekly_spearman_ic"]].subtract(base[["mcc", "brier", "mean_weekly_spearman_ic"]])
        for fold, item in delta.iterrows():
            comparisons.append({"comparison": f"INCUMBENT_ORDINAL_T2_DEV - {baseline}", "fold_id": fold, "mcc_delta": float(item.mcc), "brier_delta": float(item.brier), "weekly_ic_delta": float(item.mean_weekly_spearman_ic) if pd.notna(item.mean_weekly_spearman_ic) else None})
    metric_table.to_csv(output / "FOLD_METRICS.csv", index=False, encoding="utf-8")
    global_table.to_csv(output / "GLOBAL_METRICS.csv", index=False, encoding="utf-8")
    pd.DataFrame(comparisons).to_csv(output / "FOLD_COMPARISONS.csv", index=False, encoding="utf-8")
    summary = {
        "node_id": "WP08_T2_DUAL_BASELINE_INDEPENDENT_METRIC_READ_V1",
        "status": "PASS_DEVELOPMENT_METRICS_READ_NO_CANDIDATE_TRAINING_AUTHORIZED",
        "sealed_run": str(sealed_run),
        "sealed_manifest_sha256": sha256(manifest_path),
        "seal_hashes_verified": True,
        "sample_sha256": sha256(samples_path),
        "fold_count": len(manifest["folds"]),
        "baseline_ids": list(BASELINES),
        "best_naive_by_global_mcc": best_naive,
        "best_naive_tie": naive_winners if best_naive is None else [],
        "global_metrics": global_table.to_dict(orient="records"),
        "fresh_payloads_opened": False,
        "screening_read": False,
        "final_read": False,
        "model_trained": False,
        "gpu_used": False,
        "candidate_training_authorized": False,
        "production_replacement_allowed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "INDEPENDENT_METRIC_READ_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_root": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


