from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_row_set_sha256(frame: pd.DataFrame) -> str:
    columns = ["stock_code", "trade_date"]
    if "target_date" in frame:
        columns.append("target_date")
    rows = frame[columns].copy()
    for column in ("trade_date", "target_date"):
        if column in rows:
            rows[column] = pd.to_datetime(rows[column], errors="coerce").dt.strftime("%Y-%m-%d")
    rows = rows.sort_values(columns).reset_index(drop=True)
    return hashlib.sha256(rows.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def canonical_sample_content_sha256(frame: pd.DataFrame) -> str:
    columns = ["stock_code", "trade_date", "target_date", "target_return"]
    rows = frame[columns].copy()
    for column in ("trade_date", "target_date"):
        rows[column] = pd.to_datetime(rows[column], errors="coerce").dt.strftime("%Y-%m-%d")
    rows["target_return"] = pd.to_numeric(rows["target_return"], errors="coerce").map(
        lambda value: "" if pd.isna(value) else f"{value:.12g}"
    )
    rows = rows.sort_values(["stock_code", "trade_date", "target_date"]).reset_index(drop=True)
    return hashlib.sha256(rows.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def manifest_root_sha256(records: Iterable[dict[str, Any]]) -> str:
    normalized = sorted(
        records,
        key=lambda item: (str(item["source_id"]), str(item["relative_path"])),
    )
    return stable_json_sha256(normalized)
