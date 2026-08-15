from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RollingOriginFold:
    fold_id: str
    train_start: str
    train_end: str
    purge_start: str
    purge_end: str
    validation_start: str
    validation_end: str


def _date_text(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def generate_fold_boundaries(
    dates: pd.Index,
    fold_count: int,
    minimum_train_weeks: int,
    validation_weeks: int,
    step_weeks: int,
    purge_weeks: int,
) -> list[RollingOriginFold]:
    ordered = pd.DatetimeIndex(pd.to_datetime(dates)).drop_duplicates().sort_values()
    if any(value < 1 for value in (fold_count, minimum_train_weeks, validation_weeks, step_weeks)):
        raise ValueError("fold, train, validation, and step sizes must be positive")
    if purge_weeks < 0:
        raise ValueError("purge_weeks cannot be negative")
    folds = []
    for chronological_index in range(fold_count):
        distance_from_latest = fold_count - 1 - chronological_index
        validation_end_index = len(ordered) - 1 - distance_from_latest * step_weeks
        validation_start_index = validation_end_index - validation_weeks + 1
        train_end_index = validation_start_index - purge_weeks - 1
        if validation_start_index < 0 or train_end_index + 1 < minimum_train_weeks:
            raise ValueError("not enough development weeks for requested rolling-origin protocol")
        purge_start_index = train_end_index + 1
        purge_end_index = validation_start_index - 1
        purge_start = ordered[purge_start_index] if purge_weeks else ordered[train_end_index]
        purge_end = ordered[purge_end_index] if purge_weeks else ordered[train_end_index]
        folds.append(RollingOriginFold(
            fold_id=f"D_RO_{chronological_index + 1:02d}",
            train_start=_date_text(ordered[0]),
            train_end=_date_text(ordered[train_end_index]),
            purge_start=_date_text(purge_start),
            purge_end=_date_text(purge_end),
            validation_start=_date_text(ordered[validation_start_index]),
            validation_end=_date_text(ordered[validation_end_index]),
        ))
    validate_fold_boundaries(folds)
    return folds


def validate_fold_boundaries(folds: list[RollingOriginFold]) -> None:
    if not folds:
        raise ValueError("rolling-origin protocol requires at least one fold")
    previous_validation_end = None
    for fold in folds:
        train_end = pd.Timestamp(fold.train_end)
        purge_start = pd.Timestamp(fold.purge_start)
        purge_end = pd.Timestamp(fold.purge_end)
        validation_start = pd.Timestamp(fold.validation_start)
        validation_end = pd.Timestamp(fold.validation_end)
        if not train_end < validation_start <= validation_end:
            raise ValueError(f"invalid time order in {fold.fold_id}")
        if purge_start != train_end and not train_end < purge_start <= purge_end < validation_start:
            raise ValueError(f"invalid purge interval in {fold.fold_id}")
        if previous_validation_end is not None and validation_end <= previous_validation_end:
            raise ValueError("fold validation endpoints must advance through time")
        previous_validation_end = validation_end


def _eligible_mask(panel: pd.DataFrame, lookback_weeks: int) -> pd.Series:
    mask = panel["target_return"].notna() & panel["target_date"].notna()
    if "history_weeks_available" in panel:
        mask &= panel["history_weeks_available"].ge(lookback_weeks)
    if "cross_section_eligible" in panel:
        mask &= panel["cross_section_eligible"].fillna(False)
    return mask


def _row_set_sha256(frame: pd.DataFrame) -> str:
    rows = frame[["stock_code", "trade_date", "target_date"]].copy()
    for column in ("trade_date", "target_date"):
        rows[column] = pd.to_datetime(rows[column]).dt.strftime("%Y-%m-%d")
    rows = rows.sort_values(["stock_code", "trade_date", "target_date"])
    payload = rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def build_fold_assignments(
    panel: pd.DataFrame,
    folds: list[RollingOriginFold],
    lookback_weeks: int,
    minimum_stock_count: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["target_date"] = pd.to_datetime(work["target_date"], errors="coerce")
    eligible = _eligible_mask(work, lookback_weeks)
    assignment_frames = []
    metadata = []
    for fold in folds:
        train_end = pd.Timestamp(fold.train_end)
        validation_start = pd.Timestamp(fold.validation_start)
        validation_end = pd.Timestamp(fold.validation_end)
        train_mask = eligible & work["trade_date"].le(train_end) & work["target_date"].le(train_end)
        validation_mask = (
            eligible
            & work["trade_date"].between(validation_start, validation_end)
            & work["target_date"].le(validation_end)
        )
        train = work.loc[train_mask].copy()
        validation = work.loc[validation_mask].copy()
        if train.empty or validation.empty:
            raise ValueError(f"{fold.fold_id} has an empty train or validation sample set")
        if validation["stock_code"].nunique() < minimum_stock_count:
            raise ValueError(f"{fold.fold_id} validation stock coverage is below minimum")
        if train["target_date"].max() >= validation["trade_date"].min():
            raise ValueError(f"{fold.fold_id} train targets overlap validation observations")
        for split_name, frame in (("train", train), ("validation", validation)):
            selected = frame[["stock_code", "trade_date", "target_date", "target_return"]].copy()
            selected.insert(0, "split", split_name)
            selected.insert(0, "fold_id", fold.fold_id)
            assignment_frames.append(selected)
        metadata.append({
            **asdict(fold),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "train_stock_count": int(train["stock_code"].nunique()),
            "validation_stock_count": int(validation["stock_code"].nunique()),
            "train_row_set_sha256": _row_set_sha256(train),
            "validation_row_set_sha256": _row_set_sha256(validation),
        })
    assignments = pd.concat(assignment_frames, ignore_index=True)
    return assignments, metadata


def protocol_sha256(fold_metadata: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        fold_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
