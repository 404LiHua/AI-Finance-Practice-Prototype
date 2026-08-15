from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import manifest_root_sha256, sha256_file, stable_json_sha256  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "stage_e/configs/development_protocol_v1.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen Stage E protocol receipt.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve_path(config["output_root"]) / config["data_batch_id"]
    receipt_path = output_root / "batch_receipt.json"
    snapshot_path = resolve_path(config["frozen_snapshot_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    payload_keys = [
        "protocol_id", "data_batch_id", "development_date_ceiling", "config_sha256",
        "custody_sha256", "panel_sha256", "panel_row_set_sha256",
        "source_manifest_root_sha256", "fold_protocol_sha256", "code",
    ]
    payload = {key: receipt[key] for key in payload_keys}
    checks = {
        "processing_batch_sha256": stable_json_sha256(payload) == receipt["processing_batch_sha256"],
        "snapshot_matches_batch": snapshot["processing_batch_sha256"] == receipt["processing_batch_sha256"],
        "snapshot_fold_protocol_matches": snapshot["fold_protocol_sha256"] == receipt["fold_protocol_sha256"],
    }
    artifact_paths = {
        "source_file_manifest_sha256": output_root / "source_file_manifest.jsonl",
        "fold_assignments_sha256": output_root / "fold_assignments.csv.gz",
        "fold_summary_sha256": output_root / "fold_summary.csv",
    }
    for key, path in artifact_paths.items():
        checks[key] = sha256_file(path) == receipt["generated_artifacts"][key]

    records = [
        json.loads(line)
        for line in (output_root / "source_file_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checks["source_manifest_root_sha256"] = (
        manifest_root_sha256(records) == receipt["source_manifest_root_sha256"]
    )
    checks["source_file_count"] = len(records) == int(receipt["source_file_count"])
    failed = sorted(key for key, passed in checks.items() if not passed)
    result = {
        "protocol_id": receipt["protocol_id"],
        "data_batch_id": receipt["data_batch_id"],
        "processing_batch_sha256": receipt["processing_batch_sha256"],
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
