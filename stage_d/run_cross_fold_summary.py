from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_d.aggregation import aggregate_cross_fold  # noqa: E402
from stage_d.custody import DataCustodyGuard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate frozen fold-seed metrics uniformly.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="naive")
    parser.add_argument(
        "--custody", type=Path,
        default=REPO_ROOT / "stage_d/configs/data_custody.json",
    )
    args = parser.parse_args()
    custody_path = args.custody if args.custody.is_absolute() else REPO_ROOT / args.custody
    guard = DataCustodyGuard.from_config(custody_path, REPO_ROOT)
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    guard.assert_path_allowed(input_path, purpose="cross-fold metric input")
    guard.assert_path_allowed(output_path, purpose="cross-fold summary output")
    metrics = pd.read_csv(input_path)
    per_fold, summary, metadata = aggregate_cross_fold(metrics, baseline=args.baseline)
    output_path.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(output_path / "per_fold_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_path / "cross_fold_model_summary.csv", index=False, encoding="utf-8-sig")
    (output_path / "aggregation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
