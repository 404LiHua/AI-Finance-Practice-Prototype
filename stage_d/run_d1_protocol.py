from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_d.custody import DataCustodyGuard  # noqa: E402
from stage_d.rolling_origin import (  # noqa: E402
    build_fold_assignments,
    generate_fold_boundaries,
    protocol_sha256,
)


DEFAULT_CONFIG = REPO_ROOT / "stage_d/configs/d1_protocol.json"
DEFAULT_CUSTODY = REPO_ROOT / "stage_d/configs/data_custody.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the sealed Stage D rolling-origin protocol.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--custody", type=Path, default=DEFAULT_CUSTODY)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    custody_path = args.custody if args.custody.is_absolute() else REPO_ROOT / args.custody
    config = json.loads(config_path.read_text(encoding="utf-8"))
    guard = DataCustodyGuard.from_config(custody_path, REPO_ROOT)
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    panel_path = guard.assert_path_allowed(data_root / "panel.csv.gz", purpose="D-1 protocol")
    panel = pd.read_csv(panel_path, low_memory=False)
    guard.assert_development_frame(panel)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["target_date"] = pd.to_datetime(panel["target_date"], errors="coerce")
    development_end = pd.Timestamp(config["development_end"])
    panel = panel[panel["trade_date"].le(development_end)].copy()
    options = config["rolling_origin"]
    dates = pd.Index(sorted(panel["trade_date"].dropna().unique()))
    folds = generate_fold_boundaries(
        dates=dates,
        fold_count=int(options["fold_count"]),
        minimum_train_weeks=int(options["minimum_train_weeks"]),
        validation_weeks=int(options["validation_weeks"]),
        step_weeks=int(options["step_weeks"]),
        purge_weeks=int(options["purge_weeks"]),
    )
    assignments, fold_metadata = build_fold_assignments(
        panel=panel,
        folds=folds,
        lookback_weeks=int(config["lookback_weeks"]),
        minimum_stock_count=int(config["minimum_stock_count"]),
    )
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    guard.assert_path_allowed(output_root, purpose="D-1 output")
    output_root.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_root / "fold_assignments.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_metadata).to_csv(
        output_root / "fold_summary.csv", index=False, encoding="utf-8-sig"
    )
    protocol = {
        "protocol_id": config["protocol_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": config["evidence_class"],
        "custody_policy_id": guard.policy_id,
        "development_date_ceiling": guard.development_date_ceiling.date().isoformat(),
        "source_panel": panel_path.relative_to(REPO_ROOT).as_posix(),
        "source_panel_sha256": sha256_file(panel_path),
        "stock_count": int(panel["stock_code"].nunique()),
        "development_week_count": len(dates),
        "fold_count": len(folds),
        "folds": fold_metadata,
        "protocol_sha256": protocol_sha256(fold_metadata),
        "c4_row_level_data_read": False,
        "future_d_screening_read": False,
    }
    (output_root / "protocol_manifest.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
