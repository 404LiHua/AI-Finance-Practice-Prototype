from __future__ import annotations

"""Independent development evaluator for the sealed WP24 predictions."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef

FOLDS = tuple(f"REV2_RO_{i:02d}" for i in range(1, 7))
BAN = ("fresh", "screening", "final", "sealed_holdout")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((p - np.eye(3)[y.astype(int)]) ** 2, axis=1)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    confidence = p.max(axis=1); prediction = p.argmax(axis=1); result = 0.0
    for left, right in zip(np.linspace(0.0, 1.0, bins + 1)[:-1], np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (confidence >= left) & ((confidence < right) if right < 1.0 else (confidence <= right))
        if mask.any():
            result += float(mask.mean()) * abs(float((prediction[mask] == y[mask]).mean()) - float(confidence[mask].mean()))
    return result


def weekly_ic(frame: pd.DataFrame, prediction: str, target: str) -> tuple[float, int]:
    values = []
    for _, group in frame.groupby("trade_date", sort=True):
        group = group[[prediction, target]].dropna()
        if len(group) < 30 or group[prediction].nunique() < 2 or group[target].nunique() < 2:
            continue
        value = spearmanr(group[prediction], group[target]).statistic
        if np.isfinite(value): values.append(float(value))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def target_frame(samples_path: Path) -> pd.DataFrame:
    samples = pd.read_csv(samples_path, usecols=["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256", "target_return_h4", "target_valid"], dtype={"fold_id": str, "split_role": str, "stock_code": str, "sample_key_sha256": str})
    samples["trade_date"] = pd.to_datetime(samples["trade_date"], errors="raise").dt.normalize()
    canonical = samples[["trade_date", "stock_code", "target_return_h4", "target_valid"]].drop_duplicates(["trade_date", "stock_code"], keep="first")
    raw = pd.to_numeric(canonical["target_return_h4"], errors="coerce")
    valid = canonical["target_valid"].astype(bool) & raw.notna()
    median = raw.where(valid).groupby(canonical["trade_date"], sort=True).transform("median")
    canonical["relative_return"] = raw - median; canonical["derived_valid"] = valid & median.notna()
    canonical["ordinal_target"] = np.select([canonical.relative_return < -0.01, canonical.relative_return > 0.01], [0, 2], default=1).astype(np.int8)
    return samples.merge(canonical[["trade_date", "stock_code", "relative_return", "derived_valid", "ordinal_target"]], on=["trade_date", "stock_code"], how="left", validate="many_to_one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prediction-suffix", default="WP24")
    args = parser.parse_args()
    source = args.source_root.resolve(); prediction_root = args.prediction_root.resolve(); baseline_root = args.baseline_root.resolve(); protocol_path = args.protocol.resolve(); output = args.output_root.resolve()
    if output.exists(): raise RuntimeError(f"refusing to overwrite output: {output}")
    if any(token in str(path).lower() for path in (source, prediction_root, baseline_root, protocol_path, output) for token in BAN): raise RuntimeError("prohibited path token")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); manifest_path = prediction_root / "PREDICTION_SEAL_MANIFEST.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SEALED_PENDING_INDEPENDENT_METRIC_READ" or manifest.get("metrics_read"):
        raise RuntimeError("prediction seal is not eligible for first independent read")
    for relative, expected in manifest["prediction_sha256"].items():
        path = prediction_root / relative
        if sha256(path) != expected: raise RuntimeError(f"prediction hash mismatch: {relative}")
    samples_path = source / "data/rg1_4_materialized/samples.csv.gz"
    samples = target_frame(samples_path)
    fold_rows = []; comparison_rows = []
    for fold in FOLDS:
        prediction_path = prediction_root / "predictions_sealed" / f"{fold}_{args.prediction_suffix}.parquet"
        candidate = pd.read_parquet(prediction_path)
        if not (candidate["fold_id"] == fold).all() or not (candidate["split_role"] == "VALIDATION").all(): raise RuntimeError(f"non-validation prediction row in {fold}")
        joined = candidate.merge(samples[["sample_key_sha256", "trade_date", "stock_code", "relative_return", "derived_valid", "ordinal_target"]], on=["sample_key_sha256", "trade_date", "stock_code"], how="left", validate="one_to_one")
        if (joined.derived_valid.astype(bool) & joined.relative_return.isna()).any(): raise RuntimeError(f"target join incomplete: {fold}")
        total_validation_rows = len(joined)
        joined = joined[joined.derived_valid.astype(bool)].copy()
        if joined.empty: raise RuntimeError(f"no valid development targets in {fold}")
        baseline_path = baseline_root / "predictions_sealed" / f"{fold}_INCUMBENT_ORDINAL_T2_DEV.parquet"; baseline = pd.read_parquet(baseline_path)
        joined = joined.merge(baseline[["sample_key_sha256", "prob_down", "prob_neutral", "prob_up", "predicted_ordinal"]].rename(columns={"prob_down": "inc_prob_down", "prob_neutral": "inc_prob_neutral", "prob_up": "inc_prob_up", "predicted_ordinal": "inc_predicted_ordinal"}), on="sample_key_sha256", how="left", validate="one_to_one")
        if joined[["inc_prob_down", "inc_prob_neutral", "inc_prob_up"]].isna().any().any(): raise RuntimeError(f"incumbent join incomplete: {fold}")
        y_return = joined.relative_return.to_numpy(float); pred_return = joined.predicted_h4_relative.to_numpy(float)
        y = joined.ordinal_target.to_numpy(int); p = joined[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float); inc_p = joined[["inc_prob_down", "inc_prob_neutral", "inc_prob_up"]].to_numpy(float)
        candidate_ic, candidate_ic_count = weekly_ic(joined, "predicted_h4_relative", "relative_return")
        zero_ic, _ = weekly_ic(joined.assign(zero=0.0), "zero", "relative_return")
        train_labels = samples[(samples.fold_id == fold) & (samples.split_role == "TRAIN") & samples.derived_valid.astype(bool)].ordinal_target.to_numpy(int); prior = np.bincount(train_labels, minlength=3).astype(float); prior /= prior.sum()
        naive_p = np.repeat(prior.reshape(1, 3), len(y), axis=0)
        candidate_mcc = float(matthews_corrcoef(y, p.argmax(axis=1))); incumbent_mcc = float(matthews_corrcoef(y, inc_p.argmax(axis=1))); naive_mcc = float(matthews_corrcoef(y, naive_p.argmax(axis=1)))
        candidate_brier = brier(y, p); incumbent_brier = brier(y, inc_p); naive_brier = brier(y, naive_p)
        row = {"fold_id": fold, "rows": int(len(joined)), "validation_rows_total": int(total_validation_rows), "valid_coverage": float(len(joined) / total_validation_rows), "candidate_mae": float(np.mean(np.abs(pred_return - y_return))), "zero_mae": float(np.mean(np.abs(y_return))), "candidate_rmse": float(np.sqrt(np.mean((pred_return - y_return) ** 2))), "candidate_weekly_ic": candidate_ic, "candidate_weekly_ic_count": candidate_ic_count, "mcc": candidate_mcc, "incumbent_mcc": incumbent_mcc, "naive_prior_mcc": naive_mcc, "mcc_delta_vs_incumbent": candidate_mcc - incumbent_mcc, "brier": candidate_brier, "incumbent_brier": incumbent_brier, "naive_prior_brier": naive_brier, "brier_delta_vs_incumbent": candidate_brier - incumbent_brier, "ece10": ece(y, p)}
        fold_rows.append(row)
    fold_table = pd.DataFrame(fold_rows); regression_positive = int((fold_table.candidate_weekly_ic > 0).sum()); regression_mae_delta = float((fold_table.candidate_mae - fold_table.zero_mae).median()); worst_mcc_delta = float(fold_table.mcc_delta_vs_incumbent.min()); worst_brier_delta = float(fold_table.brier_delta_vs_incumbent.max())
    gates = {"regression_positive_ic_folds": {"value": regression_positive, "required": 5, "pass": regression_positive >= 5}, "regression_median_mae_delta_vs_zero": {"value": regression_mae_delta, "max": 0.001, "pass": regression_mae_delta <= 0.001}, "classification_positive_mcc_vs_naive_prior": {"value": int((fold_table.mcc > fold_table.naive_prior_mcc).sum()), "required": 5, "pass": int((fold_table.mcc > fold_table.naive_prior_mcc).sum()) >= 5}, "classification_worst_mcc_delta_vs_incumbent": {"value": worst_mcc_delta, "floor": -0.02, "pass": worst_mcc_delta >= -0.02}, "classification_worst_brier_delta_vs_incumbent": {"value": worst_brier_delta, "max": 0.01, "pass": worst_brier_delta <= 0.01}}
    joint_pass = all(item["pass"] for item in gates.values())
    output.mkdir(parents=True); fold_table.to_csv(output / "FOLD_METRICS.csv", index=False); fold_table.to_csv(output / "FOLD_COMPARISONS.csv", index=False)
    global_row = {"candidate_id": protocol["model"]["id"], "rows": int(fold_table.rows.sum()), "mae_mean": float(np.average(fold_table.candidate_mae, weights=fold_table.rows)), "rmse_mean": float(np.average(fold_table.candidate_rmse, weights=fold_table.rows)), "weekly_ic_mean": float(fold_table.candidate_weekly_ic.mean()), "mcc_mean": float(np.average(fold_table.mcc, weights=fold_table.rows)), "brier_mean": float(np.average(fold_table.brier, weights=fold_table.rows)), "ece10_mean": float(np.average(fold_table.ece10, weights=fold_table.rows))}; pd.DataFrame([global_row]).to_csv(output / "GLOBAL_METRICS.csv", index=False)
    decision = {"node_id": "WP24_H4_T2_MULTITASK_INDEPENDENT_DEVELOPMENT_EVALUATION", "status": "PASS_DEVELOPMENT_JOINT_GATE" if joint_pass else "FAIL_DEVELOPMENT_JOINT_GATE_RESEARCH_ONLY", "protocol_sha256": sha256(protocol_path), "prediction_seal_manifest_sha256": sha256(manifest_path), "fold_metrics": fold_rows, "gates": gates, "production_kernel": "RG_OBGNET_CONFIRMED_SAFE_V1_1", "production_replacement_allowed": False, "fresh_payloads_opened": False, "screening_read": False, "final_read": False, "future_labels_read": False, "metric_read_once": True, "created_at_utc": datetime.now(timezone.utc).isoformat()}; (output / "WP24_INDEPENDENT_EVALUATION_DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown = ["# WP24 H4/T2 多任务共享主干开发评价", "", f"状态：`{decision['status']}`", "", "本评价读取的只有冻结开发折的验证标签；未读取未来、FRESH、SCREENING 或 FINAL 标签，未修改生产注册表。", "", "| 折 | MAE | 零预测 MAE | 周 IC | MCC | incumbent MCC | Brier | incumbent Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in fold_rows: markdown.append(f"| {row['fold_id']} | {row['candidate_mae']:.6f} | {row['zero_mae']:.6f} | {row['candidate_weekly_ic']:.6f} | {row['mcc']:.6f} | {row['incumbent_mcc']:.6f} | {row['brier']:.6f} | {row['incumbent_brier']:.6f} |")
    markdown += ["", "## 门槛", "", *[f"- {name}: `{value['value']}`，通过=`{value['pass']}`" for name, value in gates.items()], "", "生产结论：即使开发门通过，也只产生研究候选；未来独立 T2 评价和生产替换仍需独立授权。"]
    (output / "WP24_INDEPENDENT_EVALUATION_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "output_root": str(output), "gates": gates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
