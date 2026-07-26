"""Machine acceptance for the Stage-E closure and best-model release."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "stock_node_gwnet_fixed_industry_l8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e_stage_closure_acceptance_v1.json")
    args = parser.parse_args()
    e6 = json.loads((REPO_ROOT / "outputs/stage_e/e6_candidate_gate_application_acceptance_v1.json").read_text(encoding="utf-8"))
    release = REPO_ROOT / "releases/e_stage_best_model_v1"
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((release / "model_metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((release / "provenance.json").read_text(encoding="utf-8"))
    dashboard = (REPO_ROOT / "dashboard.html").read_text(encoding="utf-8")
    reports = [
        REPO_ROOT / "reports/STAGE_E_FINAL_REPORT.md",
        REPO_ROOT / "reports/STAGE_E_BEST_MODEL_CONCLUSION.md",
        REPO_ROOT / "reports/PROJECT_WORK_SUMMARY_TO_DATE.md",
    ]
    artifact_hashes_valid = all(
        (release / item["path"]).is_file() and sha256_file(release / item["path"]) == item["sha256"]
        for item in manifest["artifacts"]
    )
    checkpoints = list((release / "checkpoints").rglob("*.pt"))
    checks = {
        "e6_acceptance_passed": bool(e6["passed"]),
        "unique_candidate_matches": e6["unique_candidate"] == MODEL_ID == metrics["model_id"] == provenance["model_id"],
        "all_34_hard_gates_passed": metrics["eligible"] and metrics["failed_gate_count"] == 0 and metrics["gate_count_passed"] == 34,
        "three_folds_three_seeds_packaged": len(checkpoints) == 9 and provenance["checkpoint_count"] == 9,
        "release_artifact_hashes_valid": artifact_hashes_valid,
        "stock_and_feature_contract_present": (release / "stock_order.json").is_file() and (release / "feature_schema.json").is_file(),
        "standalone_inference_and_verifier_present": (release / "inference.py").is_file() and (release / "verify_package.py").is_file(),
        "dashboard_uses_frozen_candidate": MODEL_ID in dashboard and "0.0316216251" in dashboard and "frets_pred" not in dashboard,
        "closure_reports_present": all(path.is_file() and path.stat().st_size > 500 for path in reports),
        "readme_and_changelog_updated": MODEL_ID in (REPO_ROOT / "README.md").read_text(encoding="utf-8") and "v0.5.0-stage-e" in (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        "no_new_training_or_screening": not provenance["new_training_performed"] and not provenance["screening_accessed"],
    }
    report = {
        "stage": "E final closure acceptance", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks, "unique_candidate": MODEL_ID,
        "release_manifest_root_sha256": manifest["manifest_root_sha256"],
        "release_manifest_sha256": sha256_file(release / "manifest.json"),
        "dashboard_sha256": sha256_file(REPO_ROOT / "dashboard.html"),
        "new_training_performed": False, "screening_accessed": False,
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
