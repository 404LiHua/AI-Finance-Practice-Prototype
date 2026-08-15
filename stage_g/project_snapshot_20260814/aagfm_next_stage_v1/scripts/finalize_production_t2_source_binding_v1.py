from __future__ import annotations

"""Write immutable-source package metadata after a passing CPU reconstruction."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role(relative: str) -> str:
    if relative.startswith("src/") or relative.startswith("scripts/"):
        return "source_code"
    if relative.startswith("data/rg3_daily_raw/"):
        return "historical_daily_raw_input"
    if relative in {"data/rg1_4_materialized/samples.csv.gz", "data/rg1_4_materialized/weekly_panel.csv.gz"}:
        return "historical_materializer_input"
    if relative.startswith("governance/"):
        return "frozen_governance_or_contract"
    if relative.startswith("outputs/"):
        return "availability_metadata_not_fresh_payload"
    return "source_registry_or_manifest"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-dir", required=True, type=Path)
    parser.add_argument("--dry-run-audit", required=True, type=Path)
    args = parser.parse_args()
    root = args.binding_dir.resolve()
    dry_run_path = args.dry_run_audit.resolve()
    manifest_path = root / "SOURCE_BINDING_MANIFEST_SHA256.csv"
    audit_path = root / "SOURCE_BINDING_AUDIT.json"
    receipt_path = root / "SOURCE_BINDING_RECEIPT.json"
    if any(path.exists() for path in (manifest_path, audit_path, receipt_path)):
        raise RuntimeError("refusing to overwrite source-binding metadata")
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    if dry_run["status"] != "PASS_REPRODUCIBLE_TRAIN_AND_RG3_SOURCE_BINDING_FRESH_SEALED":
        raise RuntimeError("cannot finalize without passing CPU dry-run")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        rows.append({
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "source": "user-delivered workspace inbox copy; direct external archive inspection unavailable",
            "authority": "SOURCE_BINDING_CPU_DRY_RUN_V2_PASS",
            "role": role(relative),
        })
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "node_id": "AA_GFMNET_PRODUCTION_T2_SOURCE_BINDING_V1",
        "status": "PASS_REPRODUCIBLE_TRAIN_AND_RG3_SOURCE_BINDING_FRESH_SEALED",
        "controlled_package_file_count_before_metadata": len(rows),
        "controlled_package_source_manifest_sha256": sha256(manifest_path),
        "source_provenance": {
            "delivery": "user copied source subset from external archive into workspace inbox",
            "direct_external_archive_inspection": False,
            "identity_claim": "package bytes are bound to this controlled copy; original external archive identity was not independently inspected",
        },
        "verified_preconditions": dry_run["preconditions"],
        "manifest": dry_run["manifest"],
        "daily_mirror": dry_run["daily"],
        "sample_scope": dry_run["sample_scope"],
        "rev8_dry_run": dry_run["rev8_dry_run"],
        "rg3_dry_run": dry_run["rg3_dry_run"],
        "fresh_payloads_opened": False,
        "fresh_labels_read": False,
        "fresh_reading_scripts_executed": False,
        "model_trained": False,
        "gpu_used": False,
        "production_replacement_allowed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "node_id": audit["node_id"],
        "status": audit["status"],
        "controlled_package": str(root),
        "source_binding_manifest_sha256": sha256(manifest_path),
        "source_binding_audit_sha256": sha256(audit_path),
        "fresh_payloads_opened": False,
        "fresh_labels_read": False,
        "materializers_executed": ["REV8 train-only", "RG3 historical technical and structural features"],
        "model_trained": False,
        "gpu_used": False,
        "production_assets_modified": False,
        "created_at_utc": audit["created_at_utc"],
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "binding_dir": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


