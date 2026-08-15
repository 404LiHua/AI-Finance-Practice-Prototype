from __future__ import annotations

"""Freeze the non-label production-T2 origin registry from source-bound samples."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--expected-samples-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    samples = args.samples.resolve()
    output = args.output_root.resolve()
    expected_hash = args.expected_samples_sha256.lower()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    actual_hash = sha256(samples)
    if actual_hash != expected_hash:
        raise RuntimeError(f"samples SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")

    # Deliberately restrict the read to non-label identity and split fields.
    rows = pd.read_csv(samples, usecols=["trade_date", "stock_code", "fold_id", "split_role"], dtype=str)
    required = ["trade_date", "stock_code", "fold_id", "split_role"]
    if rows.empty or rows[required].isna().any().any() or rows["stock_code"].str.strip().eq("").any():
        raise RuntimeError("invalid non-label source index")
    dates = pd.to_datetime(rows["trade_date"], errors="raise")
    if not bool((dates.dt.weekday == 4).all()):
        raise RuntimeError("source index contains a non-Friday production-T2 origin")

    registry = (
        rows.assign(trade_date=dates.dt.strftime("%Y-%m-%d"))
        .groupby("trade_date", sort=True)
        .agg(
            membership_row_count=("stock_code", "size"),
            distinct_stock_code_count=("stock_code", "nunique"),
            fold_count=("fold_id", "nunique"),
            split_role_count=("split_role", "nunique"),
        )
        .reset_index()
    )
    if registry.empty or (registry["distinct_stock_code_count"] < 300).any():
        raise RuntimeError("source index violates the minimum 300-stock T2 universe")

    output.mkdir(parents=True)
    registry_path = output / "T2_PRODUCTION_TRAIN_ORIGIN_REGISTRY_V1.csv"
    registry.to_csv(registry_path, index=False, encoding="utf-8")
    receipt = {
        "node_id": "T2_PRODUCTION_TRAIN_ORIGIN_REGISTRY_V1",
        "status": "PASS_NON_LABEL_T2_ORIGIN_REGISTRY_FROZEN",
        "source_samples_sha256": actual_hash,
        "source_columns_read": required,
        "labels_payload_read": False,
        "fresh_payload_read": False,
        "origin_count": int(len(registry)),
        "minimum_distinct_stock_code_count": int(registry["distinct_stock_code_count"].min()),
        "maximum_distinct_stock_code_count": int(registry["distinct_stock_code_count"].max()),
        "registry_sha256": sha256(registry_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "T2_PRODUCTION_TRAIN_ORIGIN_REGISTRY_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "origin_count": receipt["origin_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


