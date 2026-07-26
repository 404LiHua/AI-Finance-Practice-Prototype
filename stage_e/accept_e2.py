"""Machine-verifiable acceptance checks for Stage E-2 text views."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def check_batch(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    panel = pd.read_csv(resolve(config["paths"]["panel"]), usecols=["stock_code", "trade_date"])
    assignments = pd.read_csv(resolve(config["paths"]["fold_assignments"]))
    text_root = resolve(config["paths"]["licensed_text_root"])
    output_root = resolve(config["paths"]["output_root"])
    events_path = text_root / "cninfo_announcements.csv"
    manifest_path = text_root / "manifest.json"
    registry_path = resolve(config["paths"]["license_registry"])
    required = [events_path, manifest_path, registry_path, output_root / "metadata.json", output_root / "no_text_view.csv.gz"]
    missing = [str(path) for path in required if not path.exists()]
    result = {"batch_id": config["data_batch_id"], "passed": False, "missing": missing, "checks": {}}
    if missing:
        return result

    events = pd.read_csv(events_path)
    events["published_at"] = pd.to_datetime(events["published_at"], utc=True, errors="coerce")
    panel_codes = set(panel["stock_code"].astype(str))
    event_codes = set(events["stock_code"].astype(str))
    ceiling = pd.Timestamp(config["development_date_ceiling"], tz="Asia/Shanghai") + pd.Timedelta(days=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_batch_sha = manifest.get("batch_sha256", "")
    manifest_payload = dict(manifest)
    manifest_payload.pop("batch_sha256", None)
    manifest_files_valid = all(
        (text_root / item["path"]).exists()
        and sha256_file(text_root / item["path"]) == item["sha256"]
        and (text_root / item["path"]).stat().st_size == item["size_bytes"]
        for item in manifest.get("files", [])
    )
    no_text = pd.read_csv(output_root / "no_text_view.csv.gz")
    key = ["fold_id", "split", "stock_code", "trade_date", "target_date", "sample_row_id"]
    complete_columns = ["source_name", "source_item_id", "source_url", "stock_code", "license_id", "source_record_sha256"]
    checks = {
        "academic_scope": config.get("required_use_scope") == "academic_research",
        "stock_coverage_complete": panel_codes <= event_codes,
        "timestamps_parseable": bool(events["published_at"].notna().all()),
        "development_ceiling_respected": bool((events["published_at"].dt.tz_convert("Asia/Shanghai") < ceiling).all()),
        "required_trace_fields_complete": bool(events[complete_columns].fillna("").astype(str).apply(lambda column: column.str.strip().ne("").all()).all()),
        "extraction_success_rate_ge_95pct": manifest.get("extracted_text_rows", 0) >= 0.95 * max(1, manifest.get("selected_metadata_rows", 0)),
        "source_manifest_files_valid": manifest_files_valid,
        "source_manifest_batch_sha_valid": declared_batch_sha == stable_json_sha256(manifest_payload),
        "no_text_keys_equal_assignments": no_text[key].astype(str).equals(assignments[key].astype(str)),
        "tfidf_all_folds_present": all((output_root / "tfidf_svd" / str(fold) / "features.csv.gz").exists() for fold in sorted(assignments["fold_id"].unique())),
        "semantic_view_present": (output_root / "semantic_view.csv.gz").exists(),
    }
    if checks["semantic_view_present"]:
        semantic = pd.read_csv(output_root / "semantic_view.csv.gz")
        checks["semantic_keys_equal_assignments"] = semantic[key].astype(str).equals(assignments[key].astype(str))
        checks["semantic_dimension_positive"] = any(str(column).startswith("text_semantic_") for column in semantic.columns)
    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    checks["sealed_or_future_data_not_read"] = not metadata.get("sealed_data_read", True) and not metadata.get("future_screening_or_final_read", True)
    result.update({
        "passed": all(checks.values()), "checks": checks,
        "panel_stock_count": len(panel_codes), "event_stock_count": len(event_codes),
        "event_rows": len(events), "assignment_rows": len(assignments),
        "selected_metadata_rows": manifest.get("selected_metadata_rows"),
        "extraction_success_rate": manifest.get("extracted_text_rows", 0) / max(1, manifest.get("selected_metadata_rows", 0)),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e2_acceptance_v1.json")
    args = parser.parse_args()
    results = [check_batch(resolve(path)) for path in args.configs]
    report = {
        "stage": "E-2", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in results), "batches": results,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
