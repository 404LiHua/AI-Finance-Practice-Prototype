"""Independent acceptance for the frozen Stage F-0 protocol."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_f.protocol import sha256_file, stable_json_sha256, validate_stress_contract


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt_path = resolve(config["paths"]["freeze_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    declared_receipt_hash = receipt.pop("receipt_sha256")
    registry = config["source_registry"]
    stress = json.loads(resolve(registry["stress_contract"]["path"]).read_text(encoding="utf-8"))
    validate_stress_contract(stress)
    checks = {
        "freeze_receipt_passed": receipt["passed"],
        "freeze_receipt_hash_valid": declared_receipt_hash == stable_json_sha256(receipt),
        "config_hash_valid": receipt["config_sha256"] == sha256_file(config_path),
        "source_registry_hashes_valid": all(sha256_file(resolve(item["path"])) == item["sha256"] for item in registry.values()),
        "fold_key_receipts_complete": len(receipt["fold_key_receipts"]) == 3 and all(row["validation_valid_sample_count"] == 500 for row in receipt["fold_key_receipts"]),
        "fold_key_receipt_root_valid": receipt["fold_key_receipts_sha256"] == stable_json_sha256(receipt["fold_key_receipts"]),
        "hard_gate_hash_valid": receipt["hard_gates_sha256"] == stable_json_sha256(config["hard_gates"]),
        "eligibility_hash_valid": receipt["eligibility_logic_sha256"] == stable_json_sha256(config["eligibility_logic"]),
        "bounded_candidates_exact": receipt["bounded_f1_candidates"] == [item["id"] for item in config["f1_bounded_candidate_models"]],
        "no_training_metric_read_or_future_access": not receipt["training_executed"] and not receipt["candidate_metrics_read"] and not receipt["gan_training_executed"] and not receipt["screening_accessed"] and not receipt["final_accessed"],
        "next_action_is_f1_non_gan": receipt["next_action"] == "run F-1 bounded non-GAN controls under this unchanged protocol",
    }
    report = {
        "stage": "F-0 protocol acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks,
        "config_sha256": sha256_file(config_path),
        "freeze_receipt_sha256": sha256_file(receipt_path),
        "training_authorized_after_acceptance": all(checks.values()),
        "authorized_scope": "F-1 bounded non-GAN controls only",
        "gan_training_authorized": False,
        "screening_authorized": False, "final_authorized": False,
    }
    output = resolve(config["paths"]["acceptance_receipt"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
