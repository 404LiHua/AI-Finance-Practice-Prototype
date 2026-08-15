from __future__ import annotations

"""Independent gated evaluator for the single sealed WP09 C0 candidate."""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FOLDS = tuple(f"REV2_RO_{i:02d}" for i in range(1, 7))
BASELINES = ("NAIVE_PRIOR", "NAIVE_NEUTRAL", "INCUMBENT_ORDINAL_T2_DEV")
SAMPLES_SHA = "60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6"


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
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
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
    values = ["target_return_h4", "target_valid", "realized_volatility_8w"]
    for value in values:
        if (samples.groupby(["trade_date", "stock_code"], sort=False)[value].nunique(dropna=False) > 1).any():
            raise RuntimeError(f"noncanonical target source: {value}")
    frame = samples[["trade_date", "stock_code", *values]].drop_duplicates(["trade_date", "stock_code"], keep="first")
    variants = target_module.build_target_variants(frame)
    return variants[["trade_date", "stock_code", "T2_return", "T2_valid", "T2_label"]].rename(columns={"T2_return": "target_return_h4", "T2_valid": "target_valid", "T2_label": "ordinal_target"})


def weekly_ics(frame: pd.DataFrame, min_stocks: int) -> pd.DataFrame:
    rows = []
    valid = frame[frame.target_valid.astype(bool)].copy()
    valid["score"] = valid.prob_up - valid.prob_down
    for trade_date, group in valid.groupby("trade_date", sort=True):
        if len(group) < min_stocks or group.score.nunique() < 3 or group.target_return_h4.nunique() < 3:
            continue
        score_rank = group.score.rank(method="average").to_numpy(float)
        target_rank = group.target_return_h4.rank(method="average").to_numpy(float)
        ic = np.corrcoef(score_rank, target_rank)[0, 1]
        if np.isfinite(ic):
            rows.append({"trade_date": pd.Timestamp(trade_date), "ic": float(ic), "score_rank": score_rank, "target_rank": target_rank})
    return pd.DataFrame(rows)


def metric(frame: pd.DataFrame, min_stocks: int) -> tuple[dict, pd.DataFrame]:
    valid = frame[frame.target_valid.astype(bool)].copy()
    probability = valid[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)
    result = {"scored_rows": int(len(valid)), "mcc": mcc(valid.ordinal_target, valid.predicted_ordinal), "brier": brier(valid.ordinal_target, probability)}
    ic = weekly_ics(frame, min_stocks)
    result.update({"mean_weekly_spearman_ic": float(ic.ic.mean()) if len(ic) else None, "weekly_ic_count": int(len(ic))})
    return result, ic


def permutation_and_bootstrap(ic: pd.DataFrame, block_length: int, replications: int, permutation_seed: int, bootstrap_seed: int) -> dict:
    if len(ic) < block_length:
        return {"status": "INSUFFICIENT_WEEKS", "p_value_two_sided": 1.0, "bootstrap_ci": [None, None], "observed_mean_ic": None, "null_threshold_abs": None}
    observed = float(ic.ic.mean())
    score_ranks = ic.score_rank.tolist(); target_ranks = ic.target_rank.tolist()
    perm_rng = np.random.default_rng(permutation_seed)
    null = np.empty(replications, dtype=float)
    # One within-week target-rank shuffle per week, grouped and averaged in
    # contiguous eight-week blocks.  It preserves every week's cross-section;
    # blocks remain the unit of temporal aggregation fixed in WP09.
    blocks = [np.arange(start, min(start + block_length, len(ic))) for start in range(0, len(ic), block_length)]
    for rep in range(replications):
        block_means = []
        for indices in blocks:
            values = []
            for index in indices:
                shuffled = target_ranks[index][perm_rng.permutation(len(target_ranks[index]))]
                values.append(float(np.corrcoef(score_ranks[index], shuffled)[0, 1]))
            block_means.append(float(np.mean(values)))
        null[rep] = float(np.mean(block_means))
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (replications + 1))
    boot_rng = np.random.default_rng(bootstrap_seed)
    observed_values = ic.ic.to_numpy(float)
    starts = np.arange(len(observed_values))
    boot = np.empty(replications, dtype=float)
    for rep in range(replications):
        chosen = []
        while len(chosen) < len(observed_values):
            start = int(boot_rng.choice(starts))
            chosen.extend(observed_values[(start + offset) % len(observed_values)] for offset in range(block_length))
        boot[rep] = float(np.mean(chosen[:len(observed_values)]))
    return {"status": "PASS", "observed_mean_ic": observed, "p_value_two_sided": p_value, "bootstrap_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "null_threshold_abs": float(np.quantile(np.abs(null), 0.95)), "qualified_weeks": int(len(ic)), "block_length_weeks": block_length, "replications": replications}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve(); baseline_run = args.baseline_run.resolve(); candidate_run = args.candidate_run.resolve(); protocol_path = args.protocol.resolve(); output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    if any(token in str(path).lower() for path in (source, baseline_run, candidate_run, protocol_path, output) for token in ("fresh", "screening", "final", "sealed_holdout")):
        raise RuntimeError("prohibited path token")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_TRAINING_OR_CANDIDATE_METRIC_READ":
        raise RuntimeError("protocol not frozen")
    baseline_manifest_path = baseline_run / "PREDICTION_SEAL_MANIFEST.json"; candidate_manifest_path = candidate_run / "PREDICTION_SEAL_MANIFEST.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8")); candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if baseline_manifest.get("status") != "SEALED_PENDING_INDEPENDENT_METRIC_READ" or candidate_manifest.get("status") != "SEALED_PENDING_INDEPENDENT_METRIC_READ":
        raise RuntimeError("unsealed prediction package")
    if candidate_manifest.get("protocol_sha256") != sha256(protocol_path):
        raise RuntimeError("candidate protocol hash mismatch")
    for run, manifest in ((baseline_run, baseline_manifest), (candidate_run, candidate_manifest)):
        for relative, expected in manifest["prediction_sha256"].items():
            if sha256(run / relative) != expected:
                raise RuntimeError(f"seal identity failure: {relative}")
    samples_path = source / "data/rg1_4_materialized/samples.csv.gz"
    if sha256(samples_path) != SAMPLES_SHA:
        raise RuntimeError("sample identity mismatch")
    output.mkdir(parents=True)
    target = load_module("wp09_eval_target", source / "src/rev8_targets.py", source)
    usecols = ["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256", "target_return_h4", "target_valid", "realized_volatility_8w"]
    samples = pd.read_csv(samples_path, usecols=usecols, dtype={"fold_id": str, "split_role": str, "stock_code": str})
    samples.trade_date = pd.to_datetime(samples.trade_date, errors="raise").dt.normalize()
    labels = build_labels(samples, target)
    validation = samples[samples.split_role == "VALIDATION"][["fold_id", "trade_date", "stock_code", "sample_key_sha256"]].merge(labels, on=["trade_date", "stock_code"], how="left", validate="many_to_one")
    min_stocks = int(protocol["statistics"]["weekly_minimum_valid_stocks"])
    frames = {}
    for baseline in BASELINES:
        parts = []
        for fold in FOLDS:
            pred = pd.read_parquet(baseline_run / "predictions_sealed" / f"{fold}_{baseline}.parquet", engine="pyarrow")
            pred.trade_date = pd.to_datetime(pred.trade_date, errors="raise").dt.normalize()
            expected = validation[validation.fold_id == fold]
            joined = expected.merge(pred, on=["fold_id", "trade_date", "stock_code", "sample_key_sha256"], how="left", validate="one_to_one", indicator=True)
            if (joined._merge != "both").any() or (joined.baseline_id != baseline).any():
                raise RuntimeError(f"baseline coverage/identity failure: {fold}/{baseline}")
            parts.append(joined)
        frames[baseline] = pd.concat(parts, ignore_index=True)
    candidate_parts = []
    candidate_id = protocol["candidate"]["id"]
    for fold in FOLDS:
        pred = pd.read_parquet(candidate_run / "predictions_sealed" / f"{fold}_C0.parquet", engine="pyarrow")
        pred.trade_date = pd.to_datetime(pred.trade_date, errors="raise").dt.normalize()
        expected = validation[validation.fold_id == fold]
        joined = expected.merge(pred, on=["fold_id", "trade_date", "stock_code", "sample_key_sha256"], how="left", validate="one_to_one", indicator=True)
        if (joined._merge != "both").any() or (joined.candidate_id != candidate_id).any():
            raise RuntimeError(f"candidate coverage/identity failure: {fold}")
        candidate_parts.append(joined)
    frames[candidate_id] = pd.concat(candidate_parts, ignore_index=True)
    all_metrics = []; fold_metrics = []; weekly_by_model = {}
    for model_id, frame in frames.items():
        total, weekly = metric(frame, min_stocks); total["model_id"] = model_id; all_metrics.append(total); weekly_by_model[model_id] = weekly
        for fold in FOLDS:
            value, _ = metric(frame[frame.fold_id == fold], min_stocks); value.update({"model_id": model_id, "fold_id": fold}); fold_metrics.append(value)
    global_table = pd.DataFrame(all_metrics).sort_values("model_id", kind="mergesort"); fold_table = pd.DataFrame(fold_metrics).sort_values(["model_id", "fold_id"], kind="mergesort")
    candidate_fold = fold_table[fold_table.model_id == candidate_id].set_index("fold_id")
    gate_details = {}
    for baseline in ("NAIVE_PRIOR", "NAIVE_NEUTRAL"):
        reference = fold_table[fold_table.model_id == baseline].set_index("fold_id")
        delta = candidate_fold.mcc - reference.mcc
        gate_details[f"positive_mcc_vs_{baseline}"] = {"positive_blocks": int((delta > 0).sum()), "required": int(protocol["decision_gates"]["positive_MCC_blocks_vs_each_naive_baseline_minimum"]), "worst_delta": float(delta.min()), "pass": bool((delta > 0).sum() >= protocol["decision_gates"]["positive_MCC_blocks_vs_each_naive_baseline_minimum"])}
    gate_details["positive_ic_blocks"] = {"positive_blocks": int((candidate_fold.mean_weekly_spearman_ic > 0).sum()), "required": int(protocol["decision_gates"]["positive_IC_blocks_minimum"]), "pass": bool((candidate_fold.mean_weekly_spearman_ic > 0).sum() >= protocol["decision_gates"]["positive_IC_blocks_minimum"])}
    incumbent = fold_table[fold_table.model_id == "INCUMBENT_ORDINAL_T2_DEV"].set_index("fold_id")
    incumbent_delta = candidate_fold.mcc - incumbent.mcc
    gate_details["worst_mcc_vs_incumbent"] = {"worst_delta": float(incumbent_delta.min()), "floor": float(protocol["decision_gates"]["worst_block_MCC_delta_vs_incumbent_floor"]), "pass": bool(incumbent_delta.min() >= protocol["decision_gates"]["worst_block_MCC_delta_vs_incumbent_floor"])}
    baseline_brier = fold_table[fold_table.model_id.isin(BASELINES)].pivot(index="fold_id", columns="model_id", values="brier").min(axis=1)
    brier_delta = candidate_fold.brier - baseline_brier
    gate_details["brier_vs_best_baseline"] = {"worst_increase": float(brier_delta.max()), "max": float(protocol["decision_gates"]["Brier_increase_vs_best_baseline_max"]), "pass": bool(brier_delta.max() <= protocol["decision_gates"]["Brier_increase_vs_best_baseline_max"])}
    stats_cfg = protocol["statistics"]
    statistical = permutation_and_bootstrap(weekly_by_model[candidate_id], int(stats_cfg["permutation"]["block_length_weeks"]), int(stats_cfg["permutation"]["replications"]), int(stats_cfg["permutation"]["seed"]), int(stats_cfg["bootstrap"]["seed"]))
    # A single registered primary hypothesis is a one-element BH family.
    q_value = statistical["p_value_two_sided"]
    ci_lower = statistical["bootstrap_ci"][0]
    gate_details["statistics"] = {"p_value": q_value, "bh_q": q_value, "bh_max": float(protocol["decision_gates"]["global_BH_FDR_max"]), "bootstrap_ci": statistical["bootstrap_ci"], "ci_lower_positive": bool(ci_lower is not None and ci_lower > 0.0), "pass": bool(q_value <= protocol["decision_gates"]["global_BH_FDR_max"] and ci_lower is not None and ci_lower > 0.0)}
    passed = all(item["pass"] for item in gate_details.values())
    decision = "PASS_C0_DEVELOPMENT_GATE_ELIGIBLE_FOR_SEPARATE_FRESH_AUTHORIZATION" if passed else "FAIL_C0_DEVELOPMENT_GATE_FROZEN_NO_COMPENSATION"
    global_table.to_csv(output / "GLOBAL_METRICS.csv", index=False, encoding="utf-8")
    fold_table.to_csv(output / "FOLD_METRICS.csv", index=False, encoding="utf-8")
    comparison = fold_table[fold_table.model_id == candidate_id][["fold_id", "mcc", "brier", "mean_weekly_spearman_ic"]].rename(columns={"mcc": "candidate_mcc", "brier": "candidate_brier", "mean_weekly_spearman_ic": "candidate_ic"}).set_index("fold_id")
    for baseline in BASELINES:
        ref = fold_table[fold_table.model_id == baseline].set_index("fold_id")
        comparison[f"mcc_delta_vs_{baseline}"] = comparison.candidate_mcc - ref.mcc
        comparison[f"brier_delta_vs_{baseline}"] = comparison.candidate_brier - ref.brier
    comparison.reset_index().to_csv(output / "FOLD_COMPARISONS.csv", index=False, encoding="utf-8")
    report = {"node_id": "WP09_C0_INDEPENDENT_EVALUATION_V1", "status": decision, "protocol_sha256": sha256(protocol_path), "baseline_seal_manifest_sha256": sha256(baseline_manifest_path), "candidate_seal_manifest_sha256": sha256(candidate_manifest_path), "seal_hashes_verified": True, "global_metrics": global_table.to_dict(orient="records"), "gates": gate_details, "statistics": statistical, "fresh_payloads_opened": False, "screening_read": False, "final_read": False, "model_trained": False, "gpu_used": False, "candidate_training_authorized": False, "production_replacement_allowed": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "C0_INDEPENDENT_EVALUATION_DECISION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision, "output_root": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


