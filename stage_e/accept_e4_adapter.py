"""Machine-verifiable acceptance for the E-4.2 unified fold adapter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e4_adapter_acceptance_100stocks_v1.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = resolve(config["paths"]["output_root"])
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared_batch = metadata.get("batch_sha256", "")
    payload = dict(metadata)
    payload.pop("batch_sha256", None)
    assignments = pd.read_csv(resolve(config["paths"]["fold_assignments"]))
    fold_checks = []
    for receipt in metadata["folds"]:
        fold_id = receipt["fold_id"]
        fold_root = root / fold_id
        base_path = fold_root / "base_windows.npz"
        base = np.load(base_path)
        sample_mask = base["sample_mask"]
        sample_ids = base["sample_row_id"]
        expected_ids = set(assignments.loc[assignments["fold_id"].astype(str).eq(fold_id), "sample_row_id"].astype(str))
        actual_ids = set(sample_ids[sample_mask].astype(str))
        view_results = {}
        for view_name, expected_dim in (("no_text", 0), ("tfidf_svd", 32), ("semantic_bge", 512)):
            path = fold_root / f"text_{view_name}.npz"
            view = np.load(path)
            features = view["features"]
            unavailable = ~view["text_available"]
            view_results[view_name] = {
                "shape_valid": features.shape == (*sample_mask.shape, expected_dim),
                "sample_keys_equal": bool(np.array_equal(view["sample_row_id"], sample_ids) and np.array_equal(view["sample_mask"], sample_mask)),
                "missing_text_zero": bool(np.all(features[unavailable] == 0.0)),
                "hash_valid": sha256_file(path) == receipt["views"][view_name]["artifact_sha256"],
            }
        fold_checks.append({
            "fold_id": fold_id,
            "base_shape_valid": base["values"].shape == (receipt["cross_section_count"], int(config["lookback_weeks"]), receipt["stock_count"], len(config["feature_columns"])),
            "sample_count_preserved": int(sample_mask.sum()) == receipt["assignment_rows"] == len(expected_ids),
            "sample_ids_equal_assignments": actual_ids == expected_ids,
            "base_hash_valid": sha256_file(base_path) == receipt["base_windows_sha256"],
            "maximum_date_valid": pd.to_datetime(base["trade_date"].astype(str)).max() <= pd.Timestamp(config["development_date_ceiling"]),
            "train_and_validation_present": set(base["split"].astype(str)) == {"train", "validation"},
            "tfidf_model_hash_frozen": receipt["tfidf_train_fitted_model_sha256"] == receipt["tfidf_upstream_declared_model_sha256"],
            "views": view_results,
        })
    checks = {
        "metadata_batch_sha_valid": declared_batch == stable_json_sha256(payload),
        "all_fold_base_checks": all(all(value for key, value in fold.items() if key not in {"fold_id", "views"}) for fold in fold_checks),
        "all_view_checks": all(all(all(result.values()) for result in fold["views"].values()) for fold in fold_checks),
        "tfidf_not_refitted_in_adapter": metadata.get("tfidf_fit_policy", "").startswith("no refit"),
        "missing_text_policy_is_left_join_zero": metadata.get("text_join", "").startswith("left join"),
        "future_or_sealed_data_not_read": not metadata.get("future_or_sealed_data_read", True),
    }
    report = {
        "stage": "E-4.2", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks, "folds": fold_checks,
        "metadata_sha256": sha256_file(metadata_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
