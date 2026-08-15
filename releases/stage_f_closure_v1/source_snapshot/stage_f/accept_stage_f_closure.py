"""Independently accept the Stage-F closure archive and frozen negative conclusion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stage_e.hashing import sha256_file, stable_json_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    release = resolve(config["paths"]["release_root"])
    manifest_path = release / "SHA256_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conclusion = json.loads((release / "NEGATIVE_CONCLUSION.json").read_text(encoding="utf-8"))
    receipt = json.loads((release / "FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    f24 = json.loads(resolve(config["upstream"]["f2_4_acceptance"]["path"]).read_text(encoding="utf-8"))
    entries_valid = all(
        (release / item["path"]).is_file()
        and (release / item["path"]).stat().st_size == item["bytes"]
        and sha256_file(release / item["path"]) == item["sha256"]
        for item in manifest["entries"]
    )
    checks = {
        "f2_4_independent_acceptance_pass": f24["status"] == "PASS" and f24["eligible_candidates"] == [],
        "formal_negative_conclusion_exact": conclusion["formal_conclusion"] == config["formal_conclusion"],
        "stage_e_incumbent_retained": conclusion["retained_model"] == config["retained_model"],
        "all_four_candidate_results_retained": conclusion["candidate_gate_counts"] == config["candidate_gate_counts"],
        "gan_four_failures_non_compensable": conclusion["gan_non_compensable_stability_failures"]
        == config["gan_non_compensable_stability_failures"],
        "manifest_entries_valid": entries_valid,
        "manifest_root_matches_receipt": manifest["manifest_root_sha256"] == receipt["manifest_root_sha256"],
        "source_and_evidence_archived": receipt["source_file_count"] >= 50 and receipt["compact_evidence_file_count"] >= 100,
        "final_report_archived": any(item["path"].endswith("reports/STAGE_F_FINAL_REPORT.md") for item in manifest["entries"]),
        "negative_results_not_deleted": any("failure" in item["path"].casefold() for item in manifest["entries"]),
        "no_new_training_or_inference": not receipt["new_training_performed"] and not receipt["new_inference_performed"],
        "future_data_closed": not receipt["screening_accessed"] and not receipt["final_accessed"],
    }
    passed = all(checks.values())
    result = {
        "stage": "F final closure acceptance", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL", "checks": checks,
        "passed_checks": sum(checks.values()), "required_checks": len(checks),
        "version": config["version"], "formal_conclusion": config["formal_conclusion"],
        "retained_model": config["retained_model"], "manifest_root_sha256": manifest["manifest_root_sha256"],
        "manifest_sha256": sha256_file(manifest_path), "config_sha256": sha256_file(config_path),
        "screening_authorized": False, "final_authorized": False,
    }
    result["acceptance_sha256"] = stable_json_sha256(result)
    output = resolve(config["paths"]["acceptance"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("Stage-F closure acceptance failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(path))


if __name__ == "__main__":
    main()
