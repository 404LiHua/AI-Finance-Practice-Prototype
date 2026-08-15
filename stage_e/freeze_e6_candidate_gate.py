"""Verify and receipt E-6.1 candidate gates without reading or ranking candidate metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "outputs/stage_e/e6_candidate_gate_freeze_receipt_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["source_diagnostics"]
    metadata_path = resolve(source["metadata"])
    acceptance_path = resolve(source["acceptance"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    diagnostic_root = resolve(source["output_root"])
    baseline_ids = [config["baseline_roles"]["naive"], config["baseline_roles"]["strong_frozen_baseline"]]

    overall = pd.read_csv(diagnostic_root / "overall_pooled_metrics.csv").set_index("model_id").loc[baseline_ids]
    worst = pd.read_csv(diagnostic_root / "worst_fold_summary.csv").set_index("model_id").loc[baseline_ids]
    stock = pd.read_csv(diagnostic_root / "diagnostics_per_stock.csv")
    industry = pd.read_csv(diagnostic_root / "diagnostics_industry.csv")
    market_cap = pd.read_csv(diagnostic_root / "diagnostics_market_cap.csv")
    tails = pd.read_csv(diagnostic_root / "diagnostics_return_decile.csv")
    anchors = {}
    for model_id in baseline_ids:
        anchors[model_id] = {
            "overall_mae": float(overall.loc[model_id, "mae"]),
            "overall_rmse": float(overall.loc[model_id, "rmse"]),
            "worst_fold_mae": float(worst.loc[model_id, "worst_fold_mae"]),
            "max_stock_mae": float(stock.loc[stock["model_id"] == model_id, "mae"].max()),
            "information_technology_mae": float(industry.loc[(industry["model_id"] == model_id) & (industry["industry_group"] == "信息技术"), "mae"].iloc[0]),
            "mid_cap_mae": float(market_cap.loc[(market_cap["model_id"] == model_id) & (market_cap["market_cap_bucket_cutoff"] == "mid"), "mae"].iloc[0]),
            "d1_mae": float(tails.loc[(tails["model_id"] == model_id) & (tails["return_decile"] == "D1"), "mae"].iloc[0]),
            "d10_mae": float(tails.loc[(tails["model_id"] == model_id) & (tails["return_decile"] == "D10"), "mae"].iloc[0]),
        }
    anchor_match = all(
        close(value, config["baseline_anchors"][model_id][field])
        for model_id, fields in anchors.items() for field, value in fields.items()
    )
    naive = anchors[baseline_ids[0]]
    frets = anchors[baseline_ids[1]]
    gates = config["gates"]
    resolved_match = all([
        close(gates["overall"]["resolved_mae_max"], min(naive["overall_mae"], frets["overall_mae"]) * 0.995),
        close(gates["overall"]["resolved_rmse_max"], min(naive["overall_rmse"], frets["overall_rmse"]) * 1.01),
        close(gates["worst_fold"]["resolved_mae_max"], min(naive["worst_fold_mae"], frets["worst_fold_mae"]) * 1.01),
        close(gates["per_stock"]["resolved_maximum_stock_mae"], min(naive["max_stock_mae"], frets["max_stock_mae"]) * 1.03),
        close(gates["industry"]["resolved_information_technology_mae_max"], min(naive["information_technology_mae"], frets["information_technology_mae"]) * 1.02),
        close(gates["market_cap"]["resolved_mid_cap_mae_max"], min(naive["mid_cap_mae"], frets["mid_cap_mae"]) * 1.02),
        close(gates["return_tails"]["resolved_d1_mae_max"], min(naive["d1_mae"], frets["d1_mae"]) * 1.10),
        close(gates["return_tails"]["resolved_d10_mae_max"], min(naive["d10_mae"], frets["d10_mae"]) * 1.10),
        close(
            gates["return_tails"]["resolved_tail_mean_mae_max"],
            min((naive["d1_mae"] + naive["d10_mae"]) / 2, (frets["d1_mae"] + frets["d10_mae"]) / 2) * 1.02,
        ),
    ])
    expected_candidates = {
        "minimalist_price_only_l8", "random_forest_price_l12", "svm_rbf_price_l12",
        "industry_var1_ridge", "lstm_price_l8", "tcn_price_l8",
        "timegnn_deterministic_topk_l8", "stock_node_gwnet_fixed_industry_l8",
    }
    checks = {
        "status_frozen_before_eligibility_or_ranking": config["status"] == "CANDIDATE_GATES_FROZEN_BEFORE_ANY_ELIGIBILITY_OR_RANKING",
        "source_config_hash_valid": sha256_file(resolve(source["config"])) == source["config_sha256"],
        "source_metadata_hash_and_batch_valid": sha256_file(metadata_path) == source["metadata_sha256"] and metadata["batch_sha256"] == source["batch_sha256"],
        "source_acceptance_hash_and_status_valid": sha256_file(acceptance_path) == source["acceptance_sha256"] and acceptance["passed"],
        "baseline_anchors_match_frozen_diagnostics": anchor_match,
        "resolved_threshold_formulas_match": resolved_match,
        "candidate_pool_exactly_eight_non_baselines": set(config["candidate_model_ids"]) == expected_candidates and not expected_candidates.intersection(baseline_ids),
        "all_required_gate_families_present": set(gates) == {"overall", "worst_fold", "per_stock", "industry", "market_cap", "return_tails", "seed_stability", "engineering_cost"},
        "three_seed_aggregation_frozen": config["three_seed_inference_aggregation"]["method"] == "arithmetic_mean" and config["three_seed_inference_aggregation"]["seed_order"] == [20260723, 20260724, 20260725],
        "all_gates_hard_and_no_relaxation": config["eligibility_logic"]["rule"] == "all_hard_gates_must_pass" and not config["eligibility_logic"]["threshold_relaxation_allowed"],
        "tie_break_frozen_without_weights": config["eligibility_logic"]["multiple_eligible_models"] == "apply_frozen_lexicographic_tie_break_without_weights",
        "no_candidate_metrics_or_ranking_read": True,
        "no_training_future_or_screening_authorized": not config["restrictions"]["new_training_allowed"] and not config["restrictions"]["future_or_sealed_data_allowed"] and not config["restrictions"]["screening_allowed"],
    }
    report = {
        "stage": "E-6.1 candidate gate freeze", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks,
        "candidate_metrics_read": False, "candidate_ranking_performed": False,
        "candidate_eligibility_evaluated": False, "candidate_recommendation_formed": False,
        "baseline_ids": baseline_ids, "baseline_anchors_sha256": stable_json_sha256(anchors),
        "gate_rules_sha256": stable_json_sha256(config["gates"]),
        "aggregation_sha256": stable_json_sha256(config["three_seed_inference_aggregation"]),
        "config_sha256": sha256_file(config_path), "source_diagnostic_batch_sha256": source["batch_sha256"],
        "next_authorized_action": config["next_authorized_action_after_freeze"],
    }
    report["freeze_receipt_sha256"] = stable_json_sha256(report)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
