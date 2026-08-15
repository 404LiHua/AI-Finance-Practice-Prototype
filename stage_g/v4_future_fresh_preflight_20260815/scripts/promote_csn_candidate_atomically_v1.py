from __future__ import annotations

"""Validate independent receipts and atomically register the frozen CSN candidate.

The default is a no-write decision.  A registry write needs --apply and every binding
must pass; a rejected receipt or mismatch leaves the incumbent registry untouched.
"""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CANDIDATE_ID = "AA_GFMNET_CROSS_SECTIONAL_NEUTRALIZED_RESIDUAL_TCN_V1"
ANCHOR_ID = "RG_OBGNET_CONFIRMED_SAFE_V1_1"
AUDIT_PASS = "PASS_CANDIDATE_READY_FOR_INDEPENDENT_RECEIPTS"
TARGET_ALIGNMENT_PASS = "PASS_PRODUCTION_T2_TARGET_ALIGNMENT"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    for argument in ["--registry", "--candidate-root", "--candidate-audit", "--fresh-receipt", "--paper-receipt", "--consumption-receipt", "--paper-rule", "--scorer", "--target-alignment-gate", "--output-root"]:
        parser.add_argument(argument, type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Perform the atomic registry replacement after all checks pass.")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    registry_before_sha = sha(args.registry)
    registry = read_json(args.registry)
    candidate_audit, fresh, paper, consumption, rule, target_gate = (read_json(args.candidate_audit), read_json(args.fresh_receipt), read_json(args.paper_receipt), read_json(args.consumption_receipt), read_json(args.paper_rule), read_json(args.target_alignment_gate))
    manifest = args.candidate_root / "MODEL_MANIFEST.csv"
    specification = args.candidate_root / "MODEL_SPECIFICATION.json"
    manifest_sha, specification_sha = sha(manifest), sha(specification)
    model_rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    model_files_match = bool(model_rows) and all(Path(row["path"]).is_file() and sha(Path(row["path"])) == row["sha256"] for row in model_rows)
    shared_auth = fresh.get("authorization_id") == paper.get("authorization_id") == consumption.get("authorization_id")
    checks = {
        "incumbent_is_expected_anchor": registry.get("active_kernel", {}).get("kernel_id") == ANCHOR_ID,
        "production_target_semantics_alignment_pass": target_gate.get("status") == TARGET_ALIGNMENT_PASS and target_gate.get("candidate_id") == CANDIDATE_ID and target_gate.get("anchor_kernel_id") == ANCHOR_ID,
        "candidate_audit_pass": candidate_audit.get("status") == AUDIT_PASS and candidate_audit.get("candidate_id") == CANDIDATE_ID,
        "candidate_manifest_matches_audit": candidate_audit.get("candidate_model_manifest_sha256") == manifest_sha,
        "candidate_specification_is_expected": read_json(specification).get("candidate_id") == CANDIDATE_ID,
        "receipts_bind_frozen_candidate": fresh.get("candidate_model_manifest_sha256") == manifest_sha and paper.get("candidate_model_manifest_sha256") == manifest_sha and fresh.get("candidate_model_specification_sha256") == specification_sha and paper.get("candidate_model_specification_sha256") == specification_sha,
        "receipts_bind_frozen_scorer": args.scorer.is_file() and fresh.get("scoring_script_sha256") == sha(args.scorer) and paper.get("scoring_script_sha256") == sha(args.scorer),
        "candidate_models_hash_verified": model_files_match,
        "frozen_rule_matches_candidate_and_anchor": rule.get("candidate_id") == CANDIDATE_ID and rule.get("anchor_kernel_id") == ANCHOR_ID and rule.get("status") == "FROZEN_BEFORE_FRESH_SCORING",
        "fresh_receipt_pass": fresh.get("status") == "PASS_FRESH_SCORING" and fresh.get("candidate_id") == CANDIDATE_ID and fresh.get("anchor_kernel_id") == ANCHOR_ID,
        "paper_receipt_pass": paper.get("status") == "PASS_PAPER_TRADING" and paper.get("candidate_id") == CANDIDATE_ID and paper.get("anchor_kernel_id") == ANCHOR_ID,
        "receipt_rule_hashes_match": fresh.get("paper_rule_sha256") == sha(args.paper_rule) and paper.get("paper_rule_sha256") == sha(args.paper_rule) and paper.get("rule_sha256") == sha(args.paper_rule),
        "minimum_eight_fresh_origins": int(fresh.get("origin_weeks", 0)) >= 8 and int(paper.get("origin_weeks", 0)) >= 8,
        "sealed_label_scope_only": fresh.get("labels_read") is True and fresh.get("fresh_labels_read") is True and fresh.get("test_split_read") is False and fresh.get("used_for_tuning") is False and paper.get("used_for_tuning") is False,
        "one_shot_consumption_matches_receipts": shared_auth and consumption.get("status") == "CONSUMED" and consumption.get("fresh_scoring_sha256") == sha(args.fresh_receipt) and consumption.get("paper_trading_sha256") == sha(args.paper_receipt),
        "receipts_identify_same_custodian": bool(fresh.get("custodian_identity")) and fresh.get("custodian_identity") == paper.get("custodian_identity"),
    }
    passed = all(checks.values())
    args.output_root.mkdir(parents=True)
    decision = {"node_id": "CSN_RESIDUAL_ATOMIC_PROMOTION_GATE_V1", "status": "PASS_READY_FOR_ATOMIC_PROMOTION" if passed else "FAIL_CLOSED_PROMOTION_GATE", "apply_requested": args.apply, "checks": checks, "candidate_id": CANDIDATE_ID, "candidate_model_manifest_sha256": manifest_sha, "candidate_model_specification_sha256": specification_sha, "target_alignment_gate_sha256": sha(args.target_alignment_gate), "fresh_receipt_sha256": sha(args.fresh_receipt), "paper_receipt_sha256": sha(args.paper_receipt), "registry_before_sha256": registry_before_sha, "production_kernel_modified": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    if passed and args.apply:
        backup = args.registry.with_name(f"KERNEL_REGISTRY.before_CSN_PROMOTION.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
        backup.write_bytes(args.registry.read_bytes())
        registry["active_kernel"] = {"kernel_id": CANDIDATE_ID, "role": "production", "production_kernel_replaced": True, "model_manifest": str(manifest.resolve()), "model_manifest_sha256": manifest_sha, "model_specification": str(specification.resolve()), "model_specification_sha256": specification_sha, "promotion_receipts": {"fresh": str(args.fresh_receipt.resolve()), "paper": str(args.paper_receipt.resolve()), "authorization_id": fresh["authorization_id"]}}
        registry["candidate_kernel"] = {"kernel_id": None, "role": "no_active_candidate", "state": "PROMOTED_TO_PRODUCTION", "promoted_kernel_id": CANDIDATE_ID}
        registry.setdefault("history", []).append({"action": "promote_candidate", "kernel_id": CANDIDATE_ID, "rollback_kernel_id": ANCHOR_ID, "fresh_receipt": str(args.fresh_receipt.resolve()), "paper_receipt": str(args.paper_receipt.resolve()), "created_at_utc": datetime.now(timezone.utc).isoformat()})
        handle, temp_name = tempfile.mkstemp(prefix="KERNEL_REGISTRY.", suffix=".tmp", dir=str(args.registry.parent))
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as output:
            json.dump(registry, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_name, args.registry)
        decision["production_kernel_modified"] = True
        decision["registry_backup"] = str(backup.resolve())
        decision["registry_after_sha256"] = sha(args.registry)
    (args.output_root / "PROMOTION_DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("FAIL_CLOSED_PROMOTION_GATE")


if __name__ == "__main__":
    main()
