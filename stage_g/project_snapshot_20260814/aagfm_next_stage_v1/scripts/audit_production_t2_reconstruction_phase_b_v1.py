"""CPU-only preconsumption audit for restored production-T2 inputs.

This audit validates immutable input hashes and TRAIN-side contracts.  It
deliberately never opens the three FRESH CSV payloads: their bytes may be
hashed, but their rows, labels and metrics remain inaccessible until a
separate one-shot authorization is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ASSETS = {
    "rev8_train_target": "rev8_ro01_train_target.csv.gz",
    "rg3_features": "rg3_features.csv.gz",
    "fresh1_confirmation": "fresh1_confirmation.csv.gz",
    "fresh2_confirmation": "fresh2_confirmation.csv.gz",
    "fresh3_confirmation": "fresh3_incumbent_confirmation.csv.gz",
}
RECEIPTS = {
    "rev8_train_target": "MATERIALIZATION_RECEIPT.json",
    "rg3_features": "rg3_MATERIALIZATION_RECEIPT.json",
    "fresh1_confirmation": "fresh1_MATERIALIZATION_RECEIPT.json",
    "fresh2_confirmation": "fresh2_MATERIALIZATION_RECEIPT.json",
    "fresh3_confirmation": "fresh3_MATERIALIZATION_RECEIPT.json",
}
KEY = ["trade_date", "stock_code"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_row(root: Path, role: str, path: Path) -> dict[str, Any]:
    try:
        relative_path = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        relative_path = str(path)
    return {
        "role": role,
        "relative_path": relative_path,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def receipt_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "node_id": payload.get("node_id"),
        "status": payload.get("status"),
        "target_id": payload.get("target_id"),
        "sha256": sha256(path),
    }


def expected_hashes(model: dict[str, Any]) -> dict[str, str]:
    frozen = model.get("fit_receipt", {}).get("input_sha256", {})
    return {
        "rev8_train_target": str(frozen.get("data/rev8_materialized/rev8_ro01_train_target.csv.gz", "")),
        "rg3_features": str(frozen.get("data/rg3_materialized/rg3_features.csv.gz", "")),
        "fresh1_confirmation": str(frozen.get("data/fresh1_confirmation/fresh1_confirmation.csv.gz", "")),
        "fresh2_confirmation": str(frozen.get("data/fresh2_confirmation/fresh2_confirmation.csv.gz", "")),
        "fresh3_confirmation": str(frozen.get("data/fresh3_incumbent_confirmation/fresh3_incumbent_confirmation.csv.gz", "")),
    }


def audit_train_contract(root: Path, model: dict[str, Any]) -> dict[str, Any]:
    target = pd.read_csv(root / ASSETS["rev8_train_target"], parse_dates=["trade_date"])
    features = pd.read_csv(root / ASSETS["rg3_features"], parse_dates=["trade_date", "source_trade_date"])
    required_target = {
        *KEY,
        "raw_target_return_h4",
        "market_h4_median",
        "target_return_h4",
        "target_threshold",
        "target_valid",
        "ordinal_target",
    }
    required_features = {*KEY, "source_trade_date", *model["features"]}
    missing_target = sorted(required_target.difference(target.columns))
    missing_features = sorted(required_features.difference(features.columns))
    valid = target["target_valid"].fillna(False).astype(bool)
    formula_error = (target["raw_target_return_h4"] - target["market_h4_median"] - target["target_return_h4"]).abs()
    expected_ordinal = np.where(
        target["target_return_h4"] < -0.01,
        0,
        np.where(target["target_return_h4"] > 0.01, 2, 1),
    )
    joined = target[KEY].merge(features[KEY + ["source_trade_date"]], on=KEY, how="left", indicator=True)
    return {
        "fresh_payloads_opened": False,
        "target_rows": int(len(target)),
        "target_unique_keys": bool(not target.duplicated(KEY).any()),
        "target_valid_rows": int(valid.sum()),
        "target_date_range": [str(target.trade_date.min().date()), str(target.trade_date.max().date())],
        "target_missing_required_columns": missing_target,
        "target_formula_max_abs_error": float(formula_error[valid].max()) if valid.any() else None,
        "target_threshold_values": sorted(pd.to_numeric(target["target_threshold"], errors="coerce").dropna().unique().tolist()),
        "target_ordinal_exact_valid": bool((target.loc[valid, "ordinal_target"].to_numpy() == expected_ordinal[valid]).all()),
        "feature_rows": int(len(features)),
        "feature_unique_keys": bool(not features.duplicated(KEY).any()),
        "feature_stock_count": int(features.stock_code.astype(str).nunique()),
        "feature_date_range": [str(features.trade_date.min().date()), str(features.trade_date.max().date())],
        "feature_missing_required_columns": missing_features,
        "feature_source_after_trade_rows": int((features.source_trade_date > features.trade_date).sum()),
        "target_feature_join_counts": {str(k): int(v) for k, v in joined["_merge"].value_counts().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--restoration-report", type=Path, required=True)
    parser.add_argument("--vault-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    model = json.loads(args.model_json.read_text(encoding="utf-8"))
    if model.get("target_id") != "T2_MARKET_RELATIVE_FIXED":
        raise RuntimeError("FAIL_CLOSED_UNEXPECTED_PRODUCTION_TARGET")
    expected = expected_hashes(model)
    asset_rows = [file_row(root, role, root / name) for role, name in ASSETS.items()]
    receipt_rows = [file_row(root, f"{role}_receipt", root / name) for role, name in RECEIPTS.items()]
    receipts = {role: receipt_status(root / name) for role, name in RECEIPTS.items()}
    mismatches = [
        row["role"]
        for row in asset_rows
        if not row["exists"] or row["sha256"] != expected.get(row["role"])
    ]
    contract = audit_train_contract(root, model)
    hard_failures: list[str] = []
    if mismatches:
        hard_failures.append(f"model_hash_mismatch={mismatches}")
    if any(not row["exists"] for row in receipt_rows):
        hard_failures.append("missing_materialization_receipt")
    if contract["target_missing_required_columns"] or contract["feature_missing_required_columns"]:
        hard_failures.append("missing_train_contract_columns")
    for field in ("target_unique_keys", "feature_unique_keys", "target_ordinal_exact_valid"):
        if not contract[field]:
            hard_failures.append(field)
    if contract["target_formula_max_abs_error"] is None or contract["target_formula_max_abs_error"] > 1e-10:
        hard_failures.append("target_formula")
    if contract["target_threshold_values"] != [0.01]:
        hard_failures.append("target_threshold")
    if contract["feature_source_after_trade_rows"] != 0:
        hard_failures.append("feature_future_source")
    if contract["target_feature_join_counts"].get("both", 0) != contract["target_rows"]:
        hard_failures.append("target_feature_key_join")
    output.mkdir(parents=True, exist_ok=False)
    manifest_rows = asset_rows + receipt_rows + [
        file_row(root.parent, "production_model_json", args.model_json),
        file_row(root.parent, "restoration_report", args.restoration_report),
        file_row(root.parent, "vault_manifest", args.vault_manifest),
    ]
    pd.DataFrame(manifest_rows).to_csv(output / "INPUT_MANIFEST_SHA256.csv", index=False, encoding="utf-8")
    decision = {
        "node_id": "AA_GFMNET_PRODUCTION_T2_RECONSTRUCTION_PHASE_B_PRECONSUMPTION_V1",
        "status": "PASS_READY_FOR_SOURCE_BINDING_AND_NEW_AUTHORIZATION" if not hard_failures else "FAIL_CLOSED_PRECONSUMPTION",
        "hard_failures": hard_failures,
        "model_id": model.get("model_id"),
        "target_id": model.get("target_id"),
        "asset_hashes_match_model_json": not mismatches,
        "receipts": receipts,
        "train_contract": contract,
        "fresh_payloads_opened": False,
        "fresh_labels_read": False,
        "screening_read": False,
        "final_read": False,
        "model_trained": False,
        "gpu_used": False,
        "promotion_allowed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "PHASE_B_PRECONSUMPTION_DECISION.json", decision)
    manifest_hash = sha256(output / "INPUT_MANIFEST_SHA256.csv")
    write_json(output / "EXECUTION_RECEIPT.json", {
        "node_id": decision["node_id"],
        "status": decision["status"],
        "input_manifest_sha256": manifest_hash,
        "fresh_payloads_opened": False,
        "gpu_used": False,
    })
    output_rows = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": "phase_b_preconsumption_output",
        }
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "OUTPUT_SHA256_MANIFEST.csv"
    ]
    pd.DataFrame(output_rows).to_csv(output / "OUTPUT_SHA256_MANIFEST.csv", index=False, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if hard_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


