from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.custody import StageEDataCustodyGuard  # noqa: E402
from stage_e.hashing import (  # noqa: E402
    canonical_row_set_sha256,
    manifest_root_sha256,
    sha256_file,
    stable_json_sha256,
)
from stage_e.protocol import build_frozen_assignments, load_frozen_folds  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "stage_e/configs/development_protocol_v1.json"
DEFAULT_CUSTODY = REPO_ROOT / "stage_e/configs/data_custody_v1.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def expand_source_files(
    specs: list[dict[str, Any]], guard: StageEDataCustodyGuard
) -> list[dict[str, Any]]:
    records = []
    for spec in specs:
        root = guard.assert_path_allowed(resolve_path(spec["path"]), purpose=f"hash:{spec['source_id']}")
        if spec["kind"] == "file":
            files = [root]
            relative_root = root.parent
        elif spec["kind"] == "directory_glob":
            if not root.exists():
                if spec.get("required", True):
                    raise FileNotFoundError(root)
                continue
            files = sorted(path for path in root.glob(spec["glob"]) if path.is_file())
            relative_root = root
        else:
            raise ValueError(f"unsupported source kind: {spec['kind']}")
        if spec.get("required", True) and not files:
            raise FileNotFoundError(f"no files for source {spec['source_id']}: {root}")
        for path in files:
            guard.assert_path_allowed(path, purpose=f"hash:{spec['source_id']}")
            records.append({
                "source_id": spec["source_id"],
                "source_class": spec["source_class"],
                "hash_access_mode": "bytes_only_no_row_parse",
                "relative_path": path.relative_to(relative_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return sorted(records, key=lambda item: (item["source_id"], item["relative_path"]))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Stage E development protocol and hashes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--custody", type=Path, default=DEFAULT_CUSTODY)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    custody_path = args.custody if args.custody.is_absolute() else REPO_ROOT / args.custody
    config = json.loads(config_path.read_text(encoding="utf-8"))
    guard = StageEDataCustodyGuard.from_config(custody_path, REPO_ROOT)

    panel_path = guard.assert_path_allowed(resolve_path(config["panel_path"]), purpose="Stage E panel")
    panel = pd.read_csv(panel_path, low_memory=False)
    guard.assert_development_frame(panel)
    development_ceiling = pd.Timestamp(config["development_date_ceiling"])
    if development_ceiling != guard.development_date_ceiling:
        raise ValueError("protocol and custody development ceilings differ")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel["target_date"] = pd.to_datetime(panel["target_date"], errors="coerce")
    panel = panel[panel["trade_date"].le(development_ceiling)].copy()

    folds = load_frozen_folds(config)
    assignments, fold_metadata = build_frozen_assignments(
        panel,
        folds,
        lookback_weeks=int(config["lookback_weeks"]),
        minimum_stock_count=int(config["minimum_stock_count"]),
    )
    output_root = guard.assert_path_allowed(
        resolve_path(config["output_root"]) / config["data_batch_id"], purpose="Stage E protocol output"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    source_records = expand_source_files(config["source_files"], guard)
    source_manifest_root = manifest_root_sha256(source_records)
    source_manifest_path = output_root / "source_file_manifest.jsonl"
    source_manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in source_records),
        encoding="utf-8",
    )
    assignments_path = output_root / "fold_assignments.csv.gz"
    assignments.to_csv(
        assignments_path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    fold_summary_path = output_root / "fold_summary.csv"
    pd.DataFrame(fold_metadata).to_csv(fold_summary_path, index=False, encoding="utf-8-sig")

    code_records = []
    for value in config["protocol_code_files"]:
        path = guard.assert_path_allowed(resolve_path(value), purpose="Stage E protocol code")
        code_records.append({"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(path)})
    batch_payload = {
        "protocol_id": config["protocol_id"],
        "data_batch_id": config["data_batch_id"],
        "development_date_ceiling": config["development_date_ceiling"],
        "config_sha256": sha256_file(config_path),
        "custody_sha256": sha256_file(custody_path),
        "panel_sha256": sha256_file(panel_path),
        "panel_row_set_sha256": canonical_row_set_sha256(panel),
        "source_manifest_root_sha256": source_manifest_root,
        "fold_protocol_sha256": stable_json_sha256(fold_metadata),
        "code": code_records,
    }
    receipt = {
        **batch_payload,
        "processing_batch_sha256": stable_json_sha256(batch_payload),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "STAGE_E_SELECTION_EXPOSED_DEVELOPMENT_ONLY",
        "custody_policy_id": guard.policy_id,
        "stock_count": int(panel["stock_code"].nunique()),
        "panel_row_count": int(len(panel)),
        "source_file_count": len(source_records),
        "source_file_counts_by_class": pd.Series(
            [record["source_class"] for record in source_records]
        ).value_counts().sort_index().to_dict(),
        "fold_count": len(fold_metadata),
        "folds": fold_metadata,
        "generated_artifacts": {
            "source_file_manifest_sha256": sha256_file(source_manifest_path),
            "fold_assignments_sha256": sha256_file(assignments_path),
            "fold_summary_sha256": sha256_file(fold_summary_path),
        },
        "sealed_data_read": False,
        "future_screening_or_final_read": False,
        "expansion_policy": config["expansion_policy"],
    }
    write_json(output_root / "batch_receipt.json", receipt)
    snapshot_path = guard.assert_path_allowed(resolve_path(config["frozen_snapshot_path"]), purpose="frozen protocol snapshot")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(snapshot_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
