from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fold_key_receipt(arrays: Mapping[str, np.ndarray], fold_id: str) -> dict[str, Any]:
    required = {"sample_row_id", "split", "trade_date", "target_date", "stock_code", "sample_mask"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{fold_id} missing key arrays: {missing}")
    sample_ids = np.asarray(arrays["sample_row_id"]).astype(str)
    split = np.asarray(arrays["split"]).astype(str)
    trade_date = np.asarray(arrays["trade_date"]).astype(str)
    target_date = np.asarray(arrays["target_date"]).astype(str)
    stocks = np.asarray(arrays["stock_code"]).astype(str)
    mask = np.asarray(arrays["sample_mask"]).astype(bool)
    if sample_ids.shape != target_date.shape or sample_ids.shape != mask.shape:
        raise ValueError(f"{fold_id} sample key matrices do not share shape")
    if sample_ids.shape != (len(split), len(stocks)) or len(trade_date) != len(split):
        raise ValueError(f"{fold_id} cross-section dimensions are inconsistent")
    rows = []
    for time_index in range(len(split)):
        for stock_index, stock in enumerate(stocks):
            rows.append({
                "fold_id": fold_id,
                "split": str(split[time_index]),
                "trade_date": str(trade_date[time_index]),
                "target_date": str(target_date[time_index, stock_index]),
                "stock_code": str(stock),
                "sample_row_id": str(sample_ids[time_index, stock_index]),
                "sample_valid": bool(mask[time_index, stock_index]),
            })
    return {
        "fold_id": fold_id,
        "cross_section_count": len(split),
        "stock_count": len(stocks),
        "train_cross_section_count": int((split == "train").sum()),
        "validation_cross_section_count": int((split == "validation").sum()),
        "valid_sample_count": int(mask.sum()),
        "validation_valid_sample_count": int(mask[split == "validation"].sum()),
        "maximum_trade_date": max(trade_date.tolist()),
        "sample_key_sha256": stable_json_sha256(rows),
        "stock_order_sha256": stable_json_sha256(stocks.tolist()),
    }


def validate_stress_contract(config: dict[str, Any]) -> None:
    scenarios = config["scenarios"]
    ids = [str(item["id"]) for item in scenarios]
    if len(ids) != len(set(ids)) or "normal_unperturbed" not in ids:
        raise ValueError("stress scenario ids must be unique and include normal_unperturbed")
    observed = [item for item in scenarios if item["kind"] == "observed_group"]
    synthetic = [item for item in scenarios if item["kind"] == "synthetic_perturbation"]
    if len(observed) < 5 or len(synthetic) < 3:
        raise ValueError("Stage F requires at least five observed and three synthetic stress scenarios")
    for item in observed:
        if item.get("threshold_fit_scope") != "TRAIN_ONLY_PER_FOLD":
            raise ValueError(f"observed stress threshold is not TRAIN-only: {item['id']}")
    for item in synthetic:
        if not item.get("deterministic", False) or "seed_policy" not in item:
            raise ValueError(f"synthetic stress is not deterministic: {item['id']}")
