"""Freeze Stage F-0 robustness protocol without training or reading candidate metrics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_f.custody import StageFDataCustodyGuard
from stage_f.protocol import fold_key_receipt, sha256_file, stable_json_sha256, validate_stress_contract


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    registry = config["source_registry"]
    source_hashes_valid = all(sha256_file(resolve(item["path"])) == item["sha256"] for item in registry.values())
    custody = StageFDataCustodyGuard.from_config(resolve(registry["custody_config"]["path"]), REPO_ROOT)
    stress = json.loads(resolve(registry["stress_contract"]["path"]).read_text(encoding="utf-8"))
    validate_stress_contract(stress)
    acceptance = json.loads(resolve(registry["e6_acceptance"]["path"]).read_text(encoding="utf-8"))
    release_manifest = json.loads(resolve(registry["best_model_release_manifest"]["path"]).read_text(encoding="utf-8"))
    adapter_metadata = json.loads(resolve(registry["adapter_metadata"]["path"]).read_text(encoding="utf-8"))
    adapter_root = resolve(config["paths"]["adapter_root"])
    custody.assert_path_allowed(adapter_root, "stage_f_adapter")
    metadata_by_fold = {row["fold_id"]: row for row in adapter_metadata["folds"]}
    fold_receipts = []
    base_hashes_valid = True
    for fold_id in config["folds"]:
        base_path = adapter_root / fold_id / "base_windows.npz"
        custody.assert_path_allowed(base_path, "stage_f_fold")
        base_hashes_valid = base_hashes_valid and sha256_file(base_path) == metadata_by_fold[fold_id]["base_windows_sha256"]
        with np.load(base_path) as arrays:
            receipt = fold_key_receipt(arrays, fold_id)
            custody.assert_development_dates(arrays["trade_date"].astype(str), f"{fold_id}.trade_date")
            custody.assert_development_dates(arrays["target_date"].astype(str).reshape(-1), f"{fold_id}.target_date")
        receipt["base_windows_sha256"] = sha256_file(base_path)
        fold_receipts.append(receipt)
    checks = {
        "status_frozen_before_training_or_metric_read": config["status"] == "FROZEN_BEFORE_ANY_STAGE_F_TRAINING_OR_CANDIDATE_METRIC_READ",
        "all_source_hashes_valid": source_hashes_valid,
        "e6_unique_incumbent_valid": acceptance["passed"] and acceptance["unique_candidate"] == "stock_node_gwnet_fixed_industry_l8",
        "best_model_release_has_nine_checkpoints": release_manifest["model_id"] == acceptance["unique_candidate"] and len([item for item in release_manifest["artifacts"] if item["path"].endswith(".pt")]) == 9,
        "folds_and_seeds_unchanged": config["folds"] == ["E_RO_01", "E_RO_02", "E_RO_03"] and config["seeds"] == [20260723, 20260724, 20260725],
        "adapter_base_hashes_valid": base_hashes_valid,
        "fold_key_receipts_complete": len(fold_receipts) == 3 and all(row["validation_valid_sample_count"] == 500 for row in fold_receipts),
        "development_ceiling_enforced": all(row["maximum_trade_date"] <= config["development_date_ceiling"] for row in fold_receipts),
        "stress_contract_has_nine_bounded_scenarios": len(stress["scenarios"]) == 9,
        "exactly_three_bounded_non_gan_candidates": len(config["f1_bounded_candidate_models"]) == 3,
        "all_gates_hard_and_no_relaxation": config["eligibility_logic"]["rule"] == "all_hard_gates_must_pass" and not config["eligibility_logic"]["threshold_relaxation_allowed"],
        "gan_requires_separate_addendum": not config["f2_gan_boundary"]["gan_training_authorized_by_this_protocol"] and config["f2_gan_boundary"]["separate_f2_addendum_required"],
        "future_screening_final_forbidden": not config["restrictions"]["screening_allowed"] and not config["restrictions"]["final_allowed"],
    }
    report = {
        "stage": "F-0 extreme-regime and robustness protocol freeze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks,
        "training_executed": False, "candidate_metrics_read": False,
        "gan_training_executed": False, "screening_accessed": False, "final_accessed": False,
        "control_models": [item["id"] for item in config["control_models"]],
        "bounded_f1_candidates": [item["id"] for item in config["f1_bounded_candidate_models"]],
        "folds": config["folds"], "seeds": config["seeds"],
        "fold_key_receipts": fold_receipts,
        "fold_key_receipts_sha256": stable_json_sha256(fold_receipts),
        "hard_gates_sha256": stable_json_sha256(config["hard_gates"]),
        "eligibility_logic_sha256": stable_json_sha256(config["eligibility_logic"]),
        "stress_contract_sha256": registry["stress_contract"]["sha256"],
        "custody_config_sha256": registry["custody_config"]["sha256"],
        "config_sha256": sha256_file(config_path),
        "next_action": config["next_action_after_acceptance"],
    }
    report["receipt_sha256"] = stable_json_sha256(report)
    output = resolve(config["paths"]["freeze_receipt"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
