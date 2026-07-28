"""Run F-2.4 unified diagnostics from sealed E/F-1/F-2 predictions only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_f.run_f1_three_seed_review import stability_diagnostics
from stage_f.run_f1_unified_diagnostics import apply_gates, normal_diagnostics, stress_diagnostics


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_effective_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    requested = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = resolve(requested["extends_f1_config"]["path"])
    if sha256_file(base_path) != requested["extends_f1_config"]["sha256"]:
        raise RuntimeError("F-2.4 inherited F-1.4 configuration hash mismatch")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    gan_id = requested["gan_candidate_id"]
    config = dict(base)
    config["diagnostic_id"] = requested["diagnostic_id"]
    config["status"] = requested["status"]
    config["candidate_models"] = list(base["candidate_models"]) + [gan_id]
    config["all_models"] = list(base["all_models"]) + [gan_id]
    config["paths"] = dict(base["paths"])
    config["paths"].update(requested["paths"])
    config["source_registry"] = requested["source_registry"]
    return requested, config


def _gan_run_path(config: dict[str, Any], fold_id: str, seed: int) -> Path:
    key = "f2_2_run_root" if int(seed) == 20260725 else "f2_3_run_root"
    return resolve(config["paths"][key]) / (
        f"{fold_id}__{config['gan_candidate_id']}__seed{seed}/normal_and_stress_predictions.npz"
    )


def load_normal_predictions(requested: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    existing = pd.read_csv(resolve(config["paths"]["f1_4_normal_predictions"]), dtype={"stock_code": str})
    gan = pd.read_csv(resolve(config["paths"]["f2_3_predictions"]), dtype={"stock_code": str}).rename(
        columns={"candidate_id": "model_id"}
    )
    audit_columns = ["checkpoint_sha256", "config_sha256", "source_id"]
    for column in audit_columns:
        gan[column] = ""
    gan = gan[existing.columns]
    normal = pd.concat([existing, gan], ignore_index=True)
    expected_rows = len(config["all_models"]) * len(config["seeds"]) * len(config["folds"]) * 500
    grouped = normal.groupby(["model_id", "seed", "fold_id"]).size()
    contract = {
        "row_count": len(normal),
        "expected_row_count": expected_rows,
        "group_count": len(grouped),
        "expected_group_count": len(config["all_models"]) * len(config["seeds"]) * len(config["folds"]),
        "rows_per_group_all_500": bool(grouped.eq(500).all()),
        "model_set_exact": set(normal["model_id"].astype(str)) == set(config["all_models"]),
    }
    if contract["row_count"] != expected_rows or contract["group_count"] != contract["expected_group_count"]:
        raise RuntimeError("F-2.4 normal prediction count contract failed")
    if not contract["rows_per_group_all_500"] or not contract["model_set_exact"]:
        raise RuntimeError("F-2.4 normal prediction key/model contract failed")
    return normal, contract


def build_gan_stress_predictions(
    requested: dict[str, Any], config: dict[str, Any], gan_normal: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = {
        "normal_unperturbed", "negative_return_tail_q10", "positive_return_tail_q90",
        "high_volatility_q90", "low_liquidity_q10", "four_week_drawdown_q10",
    }
    prediction_keys = {
        "feature_noise_sigma_005": "feature_noise_prediction",
        "node_mask_10pct": "node_mask_prediction",
        "latest_week_feature_mask": "latest_week_mask_prediction",
    }
    rows: list[pd.DataFrame] = []
    checks = []
    for seed in config["seeds"]:
        for fold_id in config["folds"]:
            keys = gan_normal.loc[
                (gan_normal["seed"].astype(int) == int(seed))
                & (gan_normal["fold_id"].astype(str) == fold_id)
            ].reset_index(drop=True)
            if len(keys) != 500:
                raise RuntimeError(f"GAN normal key contract failed: {fold_id} seed={seed}")
            artifact = _gan_run_path(requested, fold_id, seed)
            with np.load(artifact) as saved:
                normal_prediction = saved["normal_prediction"].astype(float)
                target = saved["target_raw"].astype(float)
                stored_prediction = keys["prediction"].to_numpy(dtype=float).reshape(5, 100)
                stored_target = keys["target_return"].to_numpy(dtype=float).reshape(5, 100)
                prediction_difference = float(np.max(np.abs(stored_prediction - normal_prediction)))
                target_difference = float(np.max(np.abs(stored_target - target)))
                if prediction_difference > requested["normal_artifact_max_abs_difference"]:
                    raise RuntimeError("GAN saved normal prediction differs from sealed F-2.3 prediction")
                if target_difference > requested["target_artifact_max_abs_difference"]:
                    raise RuntimeError("GAN saved target differs from sealed F-2.3 target")
                checks.append({
                    "fold_id": fold_id, "seed": int(seed), "artifact_sha256": sha256_file(artifact),
                    "normal_prediction_max_abs_difference": prediction_difference,
                    "target_max_abs_difference": target_difference,
                })
                for scenario in config["stress_scenarios"]:
                    mask = saved[f"scenario_mask__{scenario}"].astype(bool)
                    prediction = normal_prediction if scenario in observed else saved[prediction_keys[scenario]].astype(float)
                    frame = keys.loc[mask.reshape(-1), [
                        "fold_id", "sample_row_id", "trade_date", "target_date", "stock_code",
                        "target_return", "sample_valid", "text_available",
                    ]].copy()
                    frame.insert(0, "scenario_id", scenario)
                    frame.insert(0, "seed", int(seed))
                    frame.insert(0, "model_id", requested["gan_candidate_id"])
                    frame["prediction"] = prediction[mask]
                    rows.append(frame)
    stress = pd.concat(rows, ignore_index=True)
    return stress, {
        "artifact_checks": checks,
        "artifact_check_count": len(checks),
        "maximum_normal_prediction_difference": max(x["normal_prediction_max_abs_difference"] for x in checks),
        "maximum_target_difference": max(x["target_max_abs_difference"] for x in checks),
        "stress_rows": len(stress),
        "new_model_inference_performed": False,
    }


def engineering_costs(requested: dict[str, Any], config: dict[str, Any], output_root: Path) -> pd.DataFrame:
    existing = pd.read_csv(resolve(config["paths"]["f1_4_engineering_costs"]))
    receipts = json.loads(resolve(config["paths"]["f2_3_receipts"]).read_text(encoding="utf-8"))
    gan_id = requested["gan_candidate_id"]
    selected = [item for item in receipts if item["candidate_id"] == gan_id]
    if len(selected) != 9:
        raise RuntimeError("F-2.4 requires all nine GAN engineering receipts")
    gan = {
        "model_id": gan_id, "role": "candidate", "run_count": len(selected),
        "total_duration_seconds": sum(float(x["duration_seconds"]) for x in selected),
        "total_training_seconds": sum(
            float(x["gan_training_seconds"]) + float(x["forecaster_training_seconds"]) for x in selected
        ),
        "maximum_parameter_count": max(
            int(x["generator_parameter_count"]) + int(x["critic_parameter_count"])
            + int(x["forecaster_parameter_count"]) for x in selected
        ),
        "independent_load_max_abs_difference": max(
            float(x["independent_load_max_abs_difference"]) for x in selected
        ),
    }
    result = pd.concat([existing, pd.DataFrame([gan])], ignore_index=True)
    result.to_csv(output_root / "engineering_costs.csv", index=False)
    return result


def run(config_path: Path, overwrite: bool = False) -> Path:
    requested, config = load_effective_config(config_path)
    if config["all_models"] != config["control_models"] + config["candidate_models"]:
        raise ValueError("F-2.4 frozen seven-model order changed")
    for source in requested["source_registry"].values():
        if sha256_file(resolve(source["path"])) != source["sha256"]:
            raise RuntimeError(f"F-2.4 source hash mismatch: {source['path']}")
    output_root = resolve(config["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    normal, normal_contract = load_normal_predictions(requested, config)
    normal_path = output_root / "unified_normal_predictions.csv.gz"
    normal.to_csv(normal_path, index=False, compression={"method": "gzip", "mtime": 0})
    universe = pd.read_csv(resolve(config["paths"]["universe_path"]), dtype={"stock_code": str})
    diagnostics = normal_diagnostics(normal, universe, config, output_root)
    stability = stability_diagnostics(
        normal.rename(columns={"model_id": "candidate_id"}), config["all_models"],
        config["folds"], config["seeds"], config["hard_gates"]["seed_stability"],
    )
    stability_path = output_root / "stability_diagnostics.json"
    stability_path.write_text(json.dumps(stability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    existing_stress = pd.read_csv(resolve(config["paths"]["f1_4_stress_predictions"]), dtype={"stock_code": str})
    gan_normal = normal.loc[normal["model_id"] == requested["gan_candidate_id"]].copy()
    gan_stress, artifact_receipt = build_gan_stress_predictions(requested, config, gan_normal)
    stress = pd.concat([existing_stress, gan_stress], ignore_index=True)
    stress_path = output_root / "unified_stress_predictions.csv.gz"
    stress.to_csv(stress_path, index=False, compression={"method": "gzip", "mtime": 0})
    stress_metrics = stress_diagnostics(stress, config, output_root)
    costs = engineering_costs(requested, config, output_root)
    gate_summary, conclusion = apply_gates(
        diagnostics, stress_metrics, stability, costs, config, output_root,
    )
    gan_stability = next(
        x for x in stability["models_in_frozen_order"] if x["candidate_id"] == requested["gan_candidate_id"]
    )
    frozen_f2_3 = json.loads(resolve(config["paths"]["f2_3_stability"]).read_text(encoding="utf-8"))
    if gan_stability["stability_gates"] != frozen_f2_3["stability_gates"]:
        raise RuntimeError("F-2.4 did not retain the sealed F-2.3 GAN stability gate result")
    if gan_stability["all_stability_gates_pass"]:
        raise RuntimeError("sealed GAN stability hard failure was lost")

    metadata = {
        "stage": "F-2.4 unified robustness diagnostics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "models_in_frozen_order": config["all_models"], "control_models": config["control_models"],
        "candidate_models": config["candidate_models"], "normal_prediction_contract": normal_contract,
        "normal_prediction_rows": len(normal), "stress_prediction_rows": len(stress),
        "gan_saved_artifact_receipt": artifact_receipt,
        "candidate_gate_summary": gate_summary.to_dict(orient="records"),
        "eligibility_conclusion": conclusion,
        "gan_stability_result": gan_stability,
        "gan_stability_hard_failure_retained": True,
        "stability_failure_compensation_allowed": False,
        "new_training_performed": False, "new_model_inference_performed": False,
        "ranking_performed": False, "candidate_deletion_performed": False,
        "threshold_relaxation_performed": False, "screening_accessed": False, "final_accessed": False,
        "config_sha256": sha256_file(config_path),
        "artifacts": {
            "normal_predictions_sha256": sha256_file(normal_path),
            "stress_predictions_sha256": sha256_file(stress_path),
            "stability_diagnostics_sha256": sha256_file(stability_path),
            "engineering_costs_sha256": sha256_file(output_root / "engineering_costs.csv"),
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
    path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(path, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
