"""Run F-1.4 unified robustness diagnostics from frozen predictions and checkpoints only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from stage_e.e5.evaluation import grouped_metric_frame, metric_row
from stage_e.e5.interface import load_fold_view, validation_key_frame
from stage_e.e5.low_cost import load_frets_module
from stage_e.e5.neural_graph import build_model
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_f.custody import StageFDataCustodyGuard
from stage_f.run_f1_three_seed_review import stability_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _checkpoint_root(stage: str, seed: int) -> Path:
    if stage == "low_cost":
        base = "e5_low_cost_baselines"
    elif stage == "neural":
        base = "e5_neural_graph_baselines"
    else:
        raise ValueError(stage)
    if int(seed) == 20260725:
        return REPO_ROOT / f"outputs/stage_e/{base}_single_seed_v1/runs"
    return REPO_ROOT / f"outputs/stage_e/{base}_three_seed_v1/seed_batches/seed_{seed}/runs"


def _control_checkpoint(stage: str, model_id: str, fold_id: str, seed: int) -> Path:
    return _checkpoint_root(stage, seed) / f"{fold_id}__{model_id}__seed{seed}" / "model.pt"


def _candidate_run_root(candidate_id: str, fold_id: str, seed: int) -> Path:
    batch = "f1_2_single_seed_engineering_v2" if int(seed) == 20260725 else "f1_3_three_seed_review_v1"
    return REPO_ROOT / "outputs/stage_f" / batch / "runs" / f"{fold_id}__{candidate_id}__seed{seed}"


def predict_frets(checkpoint: Path, values: np.ndarray, source_path: Path, module_name: str) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parameters = dict(payload["parameters"])
    if payload["family"] != "frets" or sha256_file(source_path) != payload["source_sha256"]:
        raise RuntimeError("frozen FreTS checkpoint or source mismatch")
    sequence_length = int(parameters["sequence_length"])
    x = values[:, -sequence_length:, :, 0:1].transpose(0, 2, 1, 3).reshape(-1, sequence_length, 1)
    module = load_frets_module(source_path, module_name)
    model = module.Model(SimpleNamespace(
        pred_len=1,
        enc_in=1,
        seq_len=sequence_length,
        channel_independence=str(parameters["channel_independence"]),
    ))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        scaled = model(torch.from_numpy(x.astype(np.float32)))[:, 0, 0].cpu().numpy()
    prediction = scaled * float(payload["target_std"]) + float(payload["target_mean"])
    return (prediction * float(parameters["shrinkage_alpha"])).reshape(values.shape[0], values.shape[2])


def predict_incumbent(checkpoint: Path, values: np.ndarray) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["family"] != "stock_node_gwnet_fixed_industry":
        raise RuntimeError("incumbent checkpoint family changed")
    parameters = dict(payload["parameters"])
    model = build_model(
        payload["family"],
        int(payload["input_size"]),
        int(parameters["sequence_length"]),
        parameters,
        np.asarray(payload["adjacency"], dtype=np.float32),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tensor = torch.from_numpy(values[:, -int(parameters["sequence_length"]):].astype(np.float32))
    with torch.no_grad():
        scaled = model(tensor).cpu().numpy()
    return scaled.astype(float) * float(payload["target_std"]) + float(payload["target_mean"])


def _perturbed_values(values: np.ndarray, seed: int, fold_index: int) -> dict[str, np.ndarray]:
    noise_rng = np.random.default_rng(int(seed) + 1101 + int(fold_index))
    noisy = values + noise_rng.normal(0.0, 0.05, size=values.shape).astype(np.float32)
    node_rng = np.random.default_rng(int(seed) + 2201 + int(fold_index))
    nodes = np.sort(node_rng.choice(values.shape[2], size=max(1, round(values.shape[2] * 0.10)), replace=False))
    node_masked = values.copy()
    node_masked[:, :, nodes, :] = 0.0
    latest_masked = values.copy()
    latest_masked[:, -1, :, :] = 0.0
    return {
        "feature_noise_sigma_005": noisy,
        "node_mask_10pct": node_masked,
        "latest_week_feature_mask": latest_masked,
    }


def _normal_prediction_lookup(normal: pd.DataFrame, model_id: str, seed: int, fold_id: str) -> np.ndarray:
    frame = normal.loc[
        (normal["model_id"].astype(str) == model_id)
        & (normal["seed"].astype(int) == int(seed))
        & (normal["fold_id"].astype(str) == fold_id)
    ].copy()
    frame = frame.reset_index(drop=True)
    if len(frame) != 500:
        raise RuntimeError(f"normal prediction lookup incomplete: {model_id} {seed} {fold_id}")
    return frame["prediction"].to_numpy(dtype=float).reshape(5, 100)


def build_stress_predictions(
    config: dict[str, Any], normal: pd.DataFrame, output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    reload_checks = []
    adapter_root = resolve(config["paths"]["adapter_root"])
    frets_source = resolve(config["paths"]["frets_source"])
    reference_candidate = config["candidate_models"][0]
    observed_ids = [
        "negative_return_tail_q10", "positive_return_tail_q90", "high_volatility_q90",
        "low_liquidity_q10", "four_week_drawdown_q10",
    ]
    synthetic_ids = ["feature_noise_sigma_005", "node_mask_10pct", "latest_week_feature_mask"]
    for seed in config["seeds"]:
        for fold_index, fold_id in enumerate(config["folds"]):
            view = load_fold_view(adapter_root, fold_id, "no_text")
            validation = view.split_indices("validation")
            values = view.numeric_values[validation].astype(np.float32)
            key_frame = validation_key_frame(view).reset_index(drop=True)
            target = view.target_raw[validation].astype(np.float32)
            reference_path = _candidate_run_root(reference_candidate, fold_id, seed) / "normal_and_stress_predictions.npz"
            with np.load(reference_path) as reference:
                masks = {
                    scenario: reference[f"scenario_mask__{scenario}"].astype(bool)
                    for scenario in config["stress_scenarios"]
                }
            perturbations = _perturbed_values(values, seed, fold_index)
            control_predictions: dict[str, dict[str, np.ndarray]] = {}
            naive = np.zeros_like(target, dtype=float)
            control_predictions["naive"] = {"normal_unperturbed": naive, **{name: naive for name in synthetic_ids}}
            frets_checkpoint = _control_checkpoint(
                "low_cost", "frets_return_l4__fixed_shrink_a075", fold_id, seed,
            )
            frets_normal = predict_frets(
                frets_checkpoint, values, frets_source, f"f14_frets_{fold_id}_{seed}_normal",
            )
            control_predictions["frets_return_l4__fixed_shrink_a075"] = {
                "normal_unperturbed": frets_normal,
                **{
                    scenario: predict_frets(
                        frets_checkpoint,
                        perturbations[scenario],
                        frets_source,
                        f"f14_frets_{fold_id}_{seed}_{scenario}",
                    )
                    for scenario in synthetic_ids
                },
            }
            incumbent_checkpoint = _control_checkpoint(
                "neural", "stock_node_gwnet_fixed_industry_l8", fold_id, seed,
            )
            incumbent_normal = predict_incumbent(incumbent_checkpoint, values)
            control_predictions["stock_node_gwnet_fixed_industry_l8"] = {
                "normal_unperturbed": incumbent_normal,
                **{
                    scenario: predict_incumbent(incumbent_checkpoint, perturbations[scenario])
                    for scenario in synthetic_ids
                },
            }
            for model_id in config["control_models"]:
                stored = _normal_prediction_lookup(normal, model_id, seed, fold_id)
                difference = float(np.max(np.abs(stored - control_predictions[model_id]["normal_unperturbed"])))
                if difference > float(config["normal_reload_max_abs_difference"]):
                    raise RuntimeError(f"control stress reload mismatch: {model_id} {fold_id} {seed} {difference}")
                reload_checks.append({
                    "model_id": model_id, "fold_id": fold_id, "seed": int(seed),
                    "normal_reload_max_abs_difference": difference,
                })
            for candidate_id in config["candidate_models"]:
                stress_path = _candidate_run_root(candidate_id, fold_id, seed) / "normal_and_stress_predictions.npz"
                with np.load(stress_path) as saved:
                    candidate_predictions = {
                        "normal_unperturbed": saved["normal_prediction"].astype(float),
                        "feature_noise_sigma_005": saved["feature_noise_prediction"].astype(float),
                        "node_mask_10pct": saved["node_mask_prediction"].astype(float),
                        "latest_week_feature_mask": saved["latest_week_mask_prediction"].astype(float),
                    }
                stored = _normal_prediction_lookup(normal, candidate_id, seed, fold_id)
                difference = float(np.max(np.abs(stored - candidate_predictions["normal_unperturbed"])))
                if difference > float(config["normal_reload_max_abs_difference"]):
                    raise RuntimeError(f"candidate stress reload mismatch: {candidate_id} {fold_id} {seed} {difference}")
                reload_checks.append({
                    "model_id": candidate_id, "fold_id": fold_id, "seed": int(seed),
                    "normal_reload_max_abs_difference": difference,
                })
                control_predictions[candidate_id] = candidate_predictions
            for model_id in config["all_models"]:
                predictions = control_predictions[model_id]
                for scenario in config["stress_scenarios"]:
                    source_scenario = "normal_unperturbed" if scenario in observed_ids else scenario
                    prediction = predictions[source_scenario]
                    mask = masks[scenario]
                    flat_keys = key_frame.loc[mask.reshape(-1)].copy()
                    flat_keys.insert(0, "scenario_id", scenario)
                    flat_keys.insert(0, "seed", int(seed))
                    flat_keys.insert(0, "model_id", model_id)
                    flat_keys["prediction"] = prediction[mask]
                    flat_keys["target_return"] = target[mask]
                    rows.append(flat_keys)
    stress = pd.concat(rows, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stress.to_csv(output_path, index=False, compression={"method": "gzip", "mtime": 0})
    return stress, {
        "reload_checks": reload_checks,
        "maximum_normal_reload_difference": max(item["normal_reload_max_abs_difference"] for item in reload_checks),
        "stress_prediction_rows": len(stress),
        "stress_predictions_sha256": sha256_file(output_path),
    }


def normal_diagnostics(
    predictions: pd.DataFrame, universe: pd.DataFrame, config: dict[str, Any], output_root: Path,
) -> dict[str, pd.DataFrame]:
    valid = predictions.loc[predictions["sample_valid"].astype(bool)].copy()
    mape_floor = float(config["evaluation"]["mape_floor"])
    direction_threshold = float(config["evaluation"]["direction_threshold"])
    overall = grouped_metric_frame(valid, ["model_id"], mape_floor, direction_threshold)
    fold = grouped_metric_frame(valid, ["model_id", "fold_id"], mape_floor, direction_threshold)
    per_stock = grouped_metric_frame(valid, ["model_id", "stock_code"], mape_floor, direction_threshold)
    joined = valid.merge(
        universe[["stock_code", "industry_group", "market_cap_bucket_cutoff"]],
        on="stock_code", how="left", validate="many_to_one",
    )
    joined["return_decile"] = joined.groupby(["model_id", "seed"])["target_return"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)],
        )
    )
    industry = grouped_metric_frame(joined, ["model_id", "industry_group"], mape_floor, direction_threshold)
    market = grouped_metric_frame(joined, ["model_id", "market_cap_bucket_cutoff"], mape_floor, direction_threshold)
    returns = grouped_metric_frame(joined, ["model_id", "return_decile"], mape_floor, direction_threshold)
    frames = {
        "normal_overall": overall,
        "normal_fold": fold,
        "normal_per_stock": per_stock,
        "normal_industry": industry,
        "normal_market_cap": market,
        "normal_return_decile": returns,
    }
    for name, frame in frames.items():
        frame.to_csv(output_root / f"{name}.csv", index=False)
    return frames


def stress_diagnostics(stress: pd.DataFrame, config: dict[str, Any], output_root: Path) -> pd.DataFrame:
    rows = []
    for (model_id, scenario_id), frame in stress.groupby(["model_id", "scenario_id"], sort=False):
        metrics = metric_row(
            frame,
            float(config["evaluation"]["mape_floor"]),
            float(config["evaluation"]["direction_threshold"]),
        )
        rows.append({"model_id": model_id, "scenario_id": scenario_id, **metrics})
    metrics = pd.DataFrame(rows)
    incumbent_id = config["incumbent_model"]
    incumbent = metrics.loc[metrics["model_id"] == incumbent_id, ["scenario_id", "mae"]].rename(
        columns={"mae": "incumbent_mae"}
    )
    metrics = metrics.merge(incumbent, on="scenario_id", how="left", validate="many_to_one")
    metrics["mae_ratio_vs_incumbent"] = metrics["mae"] / metrics["incumbent_mae"]
    metrics.to_csv(output_root / "stress_scenario_metrics.csv", index=False)
    return metrics


def engineering_costs(config: dict[str, Any], output_root: Path) -> pd.DataFrame:
    f1_receipts = json.loads(resolve(config["paths"]["f1_three_seed_receipts"]).read_text(encoding="utf-8"))
    control_cost = pd.read_csv(resolve(config["paths"]["stage_e_cost_summary"]))
    rows = []
    for model_id in config["control_models"]:
        source = control_cost.loc[control_cost["model_id"].astype(str) == model_id].iloc[0]
        rows.append({
            "model_id": model_id,
            "role": "control",
            "run_count": int(source["run_count"]),
            "total_duration_seconds": float(source["total_duration_seconds"]),
            "total_training_seconds": float(source["total_training_seconds"]),
            "maximum_parameter_count": int(source["maximum_parameter_count"]),
            "independent_load_max_abs_difference": float(source["independent_load_max_abs_difference"]),
        })
    for model_id in config["candidate_models"]:
        selected = [item for item in f1_receipts if item["candidate_id"] == model_id]
        rows.append({
            "model_id": model_id,
            "role": "candidate",
            "run_count": len(selected),
            "total_duration_seconds": sum(float(item["duration_seconds"]) for item in selected),
            "total_training_seconds": sum(float(item["training_seconds"]) for item in selected),
            "maximum_parameter_count": max(int(item["parameter_count"]) for item in selected),
            "independent_load_max_abs_difference": max(
                float(item["independent_load_max_abs_difference"]) for item in selected
            ),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "engineering_costs.csv", index=False)
    return frame


def apply_gates(
    normal: dict[str, pd.DataFrame],
    stress: pd.DataFrame,
    stability: dict[str, Any],
    costs: pd.DataFrame,
    config: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gates = config["hard_gates"]
    overall = normal["normal_overall"].set_index("model_id")
    fold = normal["normal_fold"]
    per_stock = normal["normal_per_stock"]
    naive_stock = per_stock.loc[per_stock["model_id"] == "naive", ["stock_code", "mae"]].rename(
        columns={"mae": "naive_mae"}
    )
    stability_by_model = {item["candidate_id"]: item for item in stability["models_in_frozen_order"]}
    cost_by_model = costs.set_index("model_id")
    non_reference = [name for name in config["stress_scenarios"] if name != "normal_unperturbed"]
    matrix_rows = []
    summary_rows = []
    for candidate_id in config["candidate_models"]:
        candidate_stock = per_stock.loc[per_stock["model_id"] == candidate_id].merge(
            naive_stock, on="stock_code", how="left", validate="one_to_one",
        )
        stock_coverage = int((candidate_stock["mae"] < candidate_stock["naive_mae"]).sum())
        maximum_stock_mae = float(candidate_stock["mae"].max())
        worst_fold_mae = float(fold.loc[fold["model_id"] == candidate_id, "mae"].max())
        candidate_stress = stress.loc[stress["model_id"] == candidate_id].set_index("scenario_id")
        ratios = candidate_stress.loc[non_reference, "mae_ratio_vs_incumbent"]
        stress_composite = float(candidate_stress.loc[non_reference, "mae"].mean())
        incumbent_composite = float(candidate_stress.loc[non_reference, "incumbent_mae"].mean())
        stability_row = stability_by_model[candidate_id]
        cost = cost_by_model.loc[candidate_id]
        checks = {
            "normal_overall_mae": float(overall.loc[candidate_id, "mae"]) <= gates["normal_performance"]["overall_mae_max"],
            "normal_worst_fold_mae": worst_fold_mae <= gates["normal_performance"]["worst_fold_mae_max"],
            "normal_stock_coverage": stock_coverage >= gates["normal_performance"]["minimum_stocks_below_naive_mae"],
            "normal_maximum_stock_mae": maximum_stock_mae <= gates["normal_performance"]["maximum_stock_mae"],
            "stress_composite": stress_composite / incumbent_composite
            <= gates["stress_performance"]["stress_composite_mae_max_ratio_vs_stage_e_incumbent"],
            "stress_scenarios_improve_1pct": int((ratios <= 0.99).sum())
            >= gates["stress_performance"]["minimum_scenarios_with_mae_improvement_at_least_1pct"],
            "stress_scenarios_not_above_incumbent": int((ratios <= 1.0).sum())
            >= gates["stress_performance"]["minimum_scenarios_with_mae_not_above_incumbent"],
            "stress_maximum_scenario_ratio": float(ratios.max())
            <= gates["stress_performance"]["maximum_any_scenario_mae_ratio_vs_incumbent"],
            "stress_negative_tail": float(candidate_stress.loc["negative_return_tail_q10", "mae_ratio_vs_incumbent"])
            <= gates["stress_performance"]["negative_tail_mae_max_ratio_vs_incumbent"],
            "stress_high_volatility": float(candidate_stress.loc["high_volatility_q90", "mae_ratio_vs_incumbent"])
            <= gates["stress_performance"]["high_volatility_mae_max_ratio_vs_incumbent"],
            "stability_seed_mae_cv": bool(stability_row["stability_gates"]["seed_mae_cv"]),
            "stability_pairwise_pearson": bool(stability_row["stability_gates"]["pairwise_pearson"]),
            "stability_pairwise_spearman": bool(stability_row["stability_gates"]["pairwise_spearman"]),
            "stability_prediction_std_mean": bool(stability_row["stability_gates"]["prediction_std_mean"]),
            "stability_prediction_std_p95": bool(stability_row["stability_gates"]["prediction_std_p95"]),
            "engineering_training_seconds": float(cost["total_training_seconds"])
            <= gates["engineering"]["total_training_seconds_max"],
            "engineering_duration_seconds": float(cost["total_duration_seconds"])
            <= gates["engineering"]["total_duration_seconds_max"],
            "engineering_parameter_count": int(cost["maximum_parameter_count"])
            <= gates["engineering"]["maximum_parameter_count"],
            "engineering_independent_load": float(cost["independent_load_max_abs_difference"])
            <= gates["engineering"]["independent_load_max_abs_difference"],
            "engineering_nine_receipts": int(cost["run_count"]) == 9,
        }
        for gate_id, passed in checks.items():
            matrix_rows.append({"candidate_id": candidate_id, "gate_id": gate_id, "passed": bool(passed)})
        summary_rows.append({
            "candidate_id": candidate_id,
            "passed_gate_count": sum(checks.values()),
            "required_gate_count": len(checks),
            "all_hard_gates_pass": all(checks.values()),
            "overall_mae": float(overall.loc[candidate_id, "mae"]),
            "worst_fold_mae": worst_fold_mae,
            "stocks_below_naive": stock_coverage,
            "maximum_stock_mae": maximum_stock_mae,
            "stress_composite_ratio": stress_composite / incumbent_composite,
            "minimum_pairwise_pearson": stability_row["minimum_pairwise_prediction_pearson"],
            "minimum_pairwise_spearman": stability_row["minimum_pairwise_prediction_spearman"],
        })
    matrix = pd.DataFrame(matrix_rows)
    summary = pd.DataFrame(summary_rows)
    eligible = summary.loc[summary["all_hard_gates_pass"], "candidate_id"].astype(str).tolist()
    if not eligible:
        conclusion = "FORMAL_NO_ROBUST_PROMOTABLE_CANDIDATE_RETAIN_STAGE_E_INCUMBENT"
        unique_candidate = None
    elif len(eligible) == 1:
        conclusion = "UNIQUE_ROBUST_CANDIDATE_RECOMMENDATION"
        unique_candidate = eligible[0]
    else:
        raise RuntimeError("multiple eligible candidates require frozen tie-break application")
    result = {
        "conclusion": conclusion,
        "eligible_candidate_count": len(eligible),
        "unique_candidate": unique_candidate,
        "stage_e_incumbent_retained": unique_candidate is None,
        "stability_failures_non_compensable": True,
        "ranking_performed": False,
        "threshold_relaxation_performed": False,
    }
    matrix.to_csv(output_root / "candidate_hard_gate_matrix.csv", index=False)
    summary.to_csv(output_root / "candidate_hard_gate_summary.csv", index=False)
    (output_root / "eligibility_conclusion.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary, result


def run(config_path: Path, overwrite: bool = False) -> Path:
    requested_config = json.loads(config_path.read_text(encoding="utf-8"))
    if "extends_config" in requested_config:
        base_path = resolve(requested_config["extends_config"]["path"])
        if sha256_file(base_path) != requested_config["extends_config"]["sha256"]:
            raise RuntimeError("F-1.4 inherited base configuration hash mismatch")
        config = json.loads(base_path.read_text(encoding="utf-8"))
        config["diagnostic_id"] = requested_config["diagnostic_id"]
        config["status"] = requested_config["status"]
        config["paths"]["output_root"] = requested_config["output_root"]
        config["source_registry"]["diagnostic_runner"] = requested_config["diagnostic_runner"]
        config["correction_receipt"] = requested_config["correction_receipt"]
    else:
        config = requested_config
    if config["all_models"] != config["control_models"] + config["candidate_models"]:
        raise ValueError("F-1.4 frozen model order changed")
    for source in config["source_registry"].values():
        if sha256_file(resolve(source["path"])) != source["sha256"]:
            raise RuntimeError(f"F-1.4 source hash mismatch: {source['path']}")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    guard = StageFDataCustodyGuard.from_config(resolve(config["paths"]["custody_config"]), REPO_ROOT)
    guard.assert_paths_allowed(
        [resolve(value) for key, value in config["paths"].items() if key != "output_root"],
        "f1_4_unified_diagnostics",
    )
    stage_e = pd.read_csv(resolve(config["paths"]["stage_e_predictions"]), dtype={"stock_code": str})
    controls = stage_e.loc[stage_e["model_id"].astype(str).isin(config["control_models"])].copy()
    candidates = pd.read_csv(resolve(config["paths"]["f1_predictions"]), dtype={"stock_code": str}).rename(
        columns={"candidate_id": "model_id"}
    )
    for column in controls.columns:
        if column not in candidates.columns:
            candidates[column] = "" if column.endswith("sha256") or column == "source_id" else False
    candidates = candidates[controls.columns]
    normal = pd.concat([controls, candidates], ignore_index=True)
    expected_rows = len(config["all_models"]) * len(config["seeds"]) * len(config["folds"]) * 500
    grouped = normal.groupby(["model_id", "seed", "fold_id"]).size()
    if len(normal) != expected_rows or len(grouped) != 54 or not grouped.eq(500).all():
        raise RuntimeError("F-1.4 normal prediction contract failed")
    if set(normal["model_id"].astype(str)) != set(config["all_models"]):
        raise RuntimeError("F-1.4 model set changed")
    normal_path = output_root / "unified_normal_predictions.csv.gz"
    normal.to_csv(normal_path, index=False, compression={"method": "gzip", "mtime": 0})
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    diagnostics = normal_diagnostics(normal, universe, config, output_root)
    stability_input = normal.rename(columns={"model_id": "candidate_id"})
    stability = stability_diagnostics(
        stability_input,
        config["all_models"],
        config["folds"],
        config["seeds"],
        config["hard_gates"]["seed_stability"],
    )
    stability_path = output_root / "stability_diagnostics.json"
    stability_path.write_text(json.dumps(stability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stress, stress_receipt = build_stress_predictions(
        config, normal, output_root / "unified_stress_predictions.csv.gz",
    )
    stress_metrics = stress_diagnostics(stress, config, output_root)
    costs = engineering_costs(config, output_root)
    gate_summary, conclusion = apply_gates(
        diagnostics, stress_metrics, stability, costs, config, output_root,
    )
    metadata = {
        "stage": "F-1.4 unified robustness diagnostics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "models_in_frozen_order": config["all_models"],
        "control_models": config["control_models"],
        "candidate_models": config["candidate_models"],
        "normal_prediction_rows": len(normal),
        "stress_prediction_rows": len(stress),
        "normal_prediction_contract_pass": True,
        "stress_inference_receipt": stress_receipt,
        "candidate_gate_summary": gate_summary.to_dict(orient="records"),
        "eligibility_conclusion": conclusion,
        "new_training_performed": False,
        "ranking_performed": False,
        "candidate_deletion_performed": False,
        "threshold_relaxation_performed": False,
        "stability_failure_compensation_allowed": False,
        "gan_training_executed": False,
        "screening_accessed": False,
        "final_accessed": False,
        "config_sha256": sha256_file(config_path),
        "artifacts": {
            "normal_predictions_sha256": sha256_file(normal_path),
            "stress_predictions_sha256": stress_receipt["stress_predictions_sha256"],
            "stability_diagnostics_sha256": sha256_file(stability_path),
            "hard_gate_matrix_sha256": sha256_file(output_root / "candidate_hard_gate_matrix.csv"),
            "hard_gate_summary_sha256": sha256_file(output_root / "candidate_hard_gate_summary.csv"),
            "eligibility_conclusion_sha256": sha256_file(output_root / "eligibility_conclusion.json"),
        },
    }
    metadata["batch_sha256"] = stable_json_sha256(metadata)
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(config_path, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
