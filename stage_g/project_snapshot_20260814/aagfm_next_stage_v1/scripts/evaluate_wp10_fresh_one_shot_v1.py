from __future__ import annotations

"""Independent, post-seal WP10 FRESH evaluator.

This program is the first authorized reader of FRESH labels.  It verifies all
prediction hashes before opening labels and is deliberately unable to train,
calibrate, choose a new candidate, edit the registry, or deploy a model.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FRESH = {
    "fresh1": "7706f1dbeebc1e065fbf55266443393a87adbf54702c4162c7d4c02841b30226",
    "fresh2": "f55914c7527df38c95fc294e29d6d9644f91c1e6e6c26ebe6184eaa39b6b61e4",
    "fresh3": "dc4b7f60157b35054a845c2223f6e50a680801a9fbbbf1bd1f371ccb4bedb688",
}
CANDIDATE_ID = "REV8_C0_TARGET_ADAPTED_HETEROSKEDASTIC_ORDINAL"
INCUMBENT_ID = "RG_OBGNET_CONFIRMED_SAFE_V1_1"
MODEL_IDS = (CANDIDATE_ID, "NAIVE_PRIOR", "NAIVE_NEUTRAL", INCUMBENT_ID)
MIN_STOCKS = 300
BLOCK_WEEKS = 8
REPLICATIONS = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mcc(target: np.ndarray, prediction: np.ndarray) -> float:
    matrix = np.zeros((3, 3), dtype=float)
    np.add.at(matrix, (np.asarray(target, dtype=int), np.asarray(prediction, dtype=int)), 1)
    total = matrix.sum(); truth = matrix.sum(axis=1); predicted = matrix.sum(axis=0)
    denominator = np.sqrt((total ** 2 - np.dot(truth, truth)) * (total ** 2 - np.dot(predicted, predicted)))
    return float((np.trace(matrix) * total - np.dot(truth, predicted)) / denominator) if denominator else 0.0


def brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.sum((probability - np.eye(3)[np.asarray(target, dtype=int)]) ** 2, axis=1)))


def calibration(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    probability = frame[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)
    max_probability = probability.max(axis=1); correct = (frame.predicted_ordinal.to_numpy(int) == frame.ordinal_target.to_numpy(int)).astype(float)
    bins = pd.qcut(pd.Series(max_probability), q=min(10, len(frame)), duplicates="drop")
    table = pd.DataFrame({"bin": bins.astype(str), "confidence": max_probability, "correct": correct}).groupby("bin", observed=True).agg(rows=("correct", "size"), mean_confidence=("confidence", "mean"), empirical_accuracy=("correct", "mean")).reset_index()
    table["absolute_gap"] = (table.mean_confidence - table.empirical_accuracy).abs()
    ece = float(np.sum(table.rows * table.absolute_gap) / len(frame)) if len(frame) else None
    return {"expected_calibration_error_10_quantile_bins": ece, "mean_max_probability": float(max_probability.mean()), "accuracy": float(correct.mean()), "bins": int(len(table))}, table


def weekly_ic(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("trade_date", sort=True):
        if len(group) < MIN_STOCKS:
            continue
        score = (group.prob_up - group.prob_down)
        target = group.target_return_h4
        if score.nunique() < 3 or target.nunique() < 3:
            continue
        sr = score.rank(method="average").to_numpy(float); tr = target.rank(method="average").to_numpy(float)
        value = float(np.corrcoef(sr, tr)[0, 1])
        if np.isfinite(value):
            rows.append({"trade_date": pd.Timestamp(date), "weekly_spearman_ic": value, "score_rank": sr, "target_rank": tr})
    return pd.DataFrame(rows)


def metric(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    valid = frame[frame.target_valid.astype(bool)].copy()
    probability = valid[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)
    calibration_summary, calibration_table = calibration(valid)
    ic = weekly_ic(valid)
    return {
        "scored_rows": int(len(valid)), "mcc": mcc(valid.ordinal_target, valid.predicted_ordinal), "brier": brier(valid.ordinal_target, probability),
        "mean_weekly_spearman_ic": float(ic.weekly_spearman_ic.mean()) if len(ic) else None, "weekly_ic_count": int(len(ic)), **calibration_summary,
    }, ic, calibration_table


def candidate_statistics(ic: pd.DataFrame, permutation_seed: int, bootstrap_seed: int) -> dict:
    if len(ic) < BLOCK_WEEKS:
        return {"status": "INSUFFICIENT_QUALIFIED_WEEKS", "qualified_weeks": int(len(ic))}
    observed_values = ic.weekly_spearman_ic.to_numpy(float); observed = float(observed_values.mean())
    ranks_score, ranks_target = ic.score_rank.tolist(), ic.target_rank.tolist()
    rng = np.random.default_rng(permutation_seed); null = np.empty(REPLICATIONS, dtype=float)
    blocks = [np.arange(start, min(start + BLOCK_WEEKS, len(ic))) for start in range(0, len(ic), BLOCK_WEEKS)]
    for rep in range(REPLICATIONS):
        block_values = []
        for indices in blocks:
            block_values.append(float(np.mean([np.corrcoef(ranks_score[index], ranks_target[index][rng.permutation(len(ranks_target[index]))])[0, 1] for index in indices])))
        null[rep] = float(np.mean(block_values))
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (REPLICATIONS + 1))
    rng = np.random.default_rng(bootstrap_seed); starts = np.arange(len(observed_values)); boot = np.empty(REPLICATIONS, dtype=float)
    for rep in range(REPLICATIONS):
        sampled: list[float] = []
        while len(sampled) < len(observed_values):
            start = int(rng.choice(starts)); sampled.extend(observed_values[(start + offset) % len(observed_values)] for offset in range(BLOCK_WEEKS))
        boot[rep] = float(np.mean(sampled[:len(observed_values)]))
    return {"status": "PASS", "qualified_weeks": int(len(ic)), "observed_mean_ic": observed, "p_value_two_sided": p_value, "moving_block_bootstrap_ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "block_length_weeks": BLOCK_WEEKS, "replications": REPLICATIONS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-root", required=True, type=Path)
    parser.add_argument("--prediction-run", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    fresh_root, run, protocol_path, output = args.fresh_root.resolve(), args.prediction_run.resolve(), args.protocol.resolve(), args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    if any(token in str(path).lower() for path in (fresh_root, run, protocol_path, output) for token in ("screening", "final", "sealed_holdout")):
        raise RuntimeError("prohibited path")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_TRAINING_OR_CANDIDATE_METRIC_READ":
        raise RuntimeError("candidate protocol is not frozen")
    manifest_path = run / "PREDICTION_SEAL_MANIFEST.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SEALED_PENDING_INDEPENDENT_FRESH_METRIC_READ" or tuple(manifest.get("model_ids", ())) != MODEL_IDS:
        raise RuntimeError("prediction package is not the required four-model sealed package")
    for relative, expected in manifest.get("prediction_sha256", {}).items():
        if sha256(run / relative) != expected:
            raise RuntimeError(f"prediction seal mismatch: {relative}")
    if len(manifest["prediction_sha256"]) != len(FRESH) * len(MODEL_IDS):
        raise RuntimeError("prediction package count mismatch")
    fresh_paths = {"fresh1": fresh_root / "fresh1_confirmation.csv.gz", "fresh2": fresh_root / "fresh2_confirmation.csv.gz", "fresh3": fresh_root / "fresh3_incumbent_confirmation.csv.gz"}
    for name, path in fresh_paths.items():
        if not path.is_file() or sha256(path) != FRESH[name]:
            raise RuntimeError(f"FRESH identity mismatch: {name}")
    # This is the authorized first FRESH-label read, strictly after every seal
    # hash passed above.
    labels_by_window = {}
    for name, path in fresh_paths.items():
        labels = pd.read_csv(path, usecols=["trade_date", "stock_code", "sample_key_sha256", "target_valid", "ordinal_target", "target_return_h4"], dtype={"stock_code": str})
        labels.trade_date = pd.to_datetime(labels.trade_date, errors="raise").dt.normalize()
        if labels.sample_key_sha256.duplicated().any():
            raise RuntimeError(f"duplicate FRESH labels keys: {name}")
        if labels.loc[labels.target_valid.astype(bool), ["ordinal_target", "target_return_h4"]].isna().any().any():
            raise RuntimeError(f"invalid FRESH label contract: {name}")
        labels_by_window[name] = labels
    output.mkdir(parents=True)
    metrics, ic_tables, calibration_tables, frames = [], [], [], {}
    for name, labels in labels_by_window.items():
        for model_id in MODEL_IDS:
            prediction = pd.read_parquet(run / "predictions_sealed" / f"{name}_{model_id}.parquet", engine="pyarrow")
            prediction.trade_date = pd.to_datetime(prediction.trade_date, errors="raise").dt.normalize()
            joined = labels.merge(prediction, on=["trade_date", "stock_code", "sample_key_sha256"], how="left", validate="one_to_one", indicator=True)
            if (joined._merge != "both").any() or (joined.model_id != model_id).any():
                raise RuntimeError(f"prediction coverage/identity failure: {name}/{model_id}")
            if not np.isfinite(joined[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)).all() or not np.allclose(joined[["prob_down", "prob_neutral", "prob_up"]].sum(axis=1), 1.0, atol=1e-12):
                raise RuntimeError(f"prediction probability failure: {name}/{model_id}")
            frames[(name, model_id)] = joined.drop(columns="_merge")
            value, ic, calibration_table = metric(frames[(name, model_id)])
            value.update({"window": name, "model_id": model_id}); metrics.append(value)
            if len(ic):
                ic.assign(window=name, model_id=model_id).drop(columns=["score_rank", "target_rank"]).to_csv(output / f"{name}_{model_id}_WEEKLY_IC.csv", index=False)
                ic_tables.append((name, model_id, ic))
            calibration_table.assign(window=name, model_id=model_id).to_csv(output / f"{name}_{model_id}_CALIBRATION.csv", index=False)
    for model_id in MODEL_IDS:
        combined = pd.concat([frames[(name, model_id)] for name in FRESH], ignore_index=True)
        value, ic, calibration_table = metric(combined)
        value.update({"window": "combined", "model_id": model_id}); metrics.append(value)
        if len(ic):
            ic.assign(window="combined", model_id=model_id).drop(columns=["score_rank", "target_rank"]).to_csv(output / f"combined_{model_id}_WEEKLY_IC.csv", index=False)
            ic_tables.append(("combined", model_id, ic))
        calibration_table.assign(window="combined", model_id=model_id).to_csv(output / f"combined_{model_id}_CALIBRATION.csv", index=False)
    metric_table = pd.DataFrame(metrics).sort_values(["window", "model_id"], kind="mergesort"); metric_table.to_csv(output / "FRESH_METRICS.csv", index=False)
    candidate_blocks = []
    for name in (*FRESH.keys(), "combined"):
        candidate = frames[(name, CANDIDATE_ID)] if name != "combined" else pd.concat([frames[(item, CANDIDATE_ID)] for item in FRESH], ignore_index=True)
        candidate_dates = sorted(candidate.trade_date.unique())
        for block_number, dates in enumerate([candidate_dates[start:start + BLOCK_WEEKS] for start in range(0, len(candidate_dates), BLOCK_WEEKS)], start=1):
            candidate_metric, _, _ = metric(candidate[candidate.trade_date.isin(dates)])
            row = {"window": name, "block": block_number, "start": pd.Timestamp(dates[0]).date().isoformat(), "end": pd.Timestamp(dates[-1]).date().isoformat(), "candidate_mcc": candidate_metric["mcc"], "candidate_brier": candidate_metric["brier"]}
            for baseline in ("NAIVE_PRIOR", "NAIVE_NEUTRAL"):
                reference = frames[(name, baseline)] if name != "combined" else pd.concat([frames[(item, baseline)] for item in FRESH], ignore_index=True)
                reference_metric, _, _ = metric(reference[reference.trade_date.isin(dates)])
                row[f"mcc_delta_vs_{baseline}"] = candidate_metric["mcc"] - reference_metric["mcc"]
                row[f"brier_delta_vs_{baseline}"] = candidate_metric["brier"] - reference_metric["brier"]
            candidate_blocks.append(row)
    pd.DataFrame(candidate_blocks).to_csv(output / "C0_VS_NAIVE_8_WEEK_BLOCKS.csv", index=False)
    candidate_stats = {}
    for window, model_id, ic in ic_tables:
        if model_id == CANDIDATE_ID:
            candidate_stats[window] = candidate_statistics(ic, 2026081409, 2026081410)
    report = {
        "node_id": "WP10_C0_INDEPENDENT_FRESH_EVALUATION_V1",
        "status": "FRESH_METRICS_REPORTED_NO_PRODUCTION_SELECTION_AUTHORIZED_RETAIN_INCUMBENT",
        "prediction_seal_manifest_sha256": sha256(manifest_path), "prediction_hashes_verified": True,
        "fresh_input_sha256": {name: sha256(path) for name, path in fresh_paths.items()}, "protocol_sha256": sha256(protocol_path),
        "metrics": metric_table.to_dict(orient="records"), "candidate_ic_statistics": candidate_stats,
        "incumbent_selection_status": "INCUMBENT_PRODUCTION_IN_SAMPLE_REFERENCE_NOT_VALID_FOR_SELECTION",
        "promotion_decision": "RETAIN_INCUMBENT",
        "promotion_rationale": ["C0 and naive predictions are one-shot FRESH results, but the incumbent was fitted using all three FRESH windows and is not a fair replacement comparator.", "The WP10 authorization does not pre-register a fair later incumbent holdout or authorize production replacement. FRESH evidence must not be used for retraining, tuning, candidate changes, calibration changes, or threshold changes."],
        "fresh_labels_read": True, "fresh_metrics_read": True, "model_trained": False, "gpu_used": False, "production_assets_modified": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "C0_INDEPENDENT_FRESH_EVALUATION_DECISION.json", report)
    print(json.dumps({"status": report["status"], "output_root": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


