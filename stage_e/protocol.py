from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stage_e.hashing import canonical_row_set_sha256, canonical_sample_content_sha256


@dataclass(frozen=True)
class FrozenFold:
    fold_id: str
    train_start_policy: str
    train_end: str
    purge_start: str
    purge_end: str
    validation_start: str
    validation_end: str


def load_frozen_folds(config: dict[str, Any]) -> list[FrozenFold]:
    folds = [FrozenFold(**item) for item in config["folds"]]
    if not folds:
        raise ValueError("Stage E requires at least one frozen fold")
    previous_validation_end = None
    for fold in folds:
        if fold.train_start_policy != "available_history_start":
            raise ValueError(f"unsupported train_start_policy in {fold.fold_id}")
        train_end = pd.Timestamp(fold.train_end)
        purge_start = pd.Timestamp(fold.purge_start)
        purge_end = pd.Timestamp(fold.purge_end)
        validation_start = pd.Timestamp(fold.validation_start)
        validation_end = pd.Timestamp(fold.validation_end)
        if not train_end < purge_start <= purge_end < validation_start <= validation_end:
            raise ValueError(f"invalid frozen date order in {fold.fold_id}")
        if previous_validation_end is not None and validation_end <= previous_validation_end:
            raise ValueError("validation endpoints must advance")
        previous_validation_end = validation_end
    return folds


def eligible_mask(panel: pd.DataFrame, lookback_weeks: int) -> pd.Series:
    required = {"target_return", "target_date", "model_eligible_pit"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel_v2 missing rolling eligibility columns: {missing}")
    mask = (
        panel["target_return"].notna()
        & panel["target_date"].notna()
        & panel["model_eligible_pit"].fillna(False)
    )
    if "history_weeks_available" in panel:
        mask &= panel["history_weeks_available"].ge(lookback_weeks)
    if "cross_section_eligible" in panel:
        mask &= panel["cross_section_eligible"].fillna(False)
    return mask


def build_frozen_assignments(
    panel: pd.DataFrame,
    folds: list[FrozenFold],
    lookback_weeks: int,
    minimum_stock_count: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["target_date"] = pd.to_datetime(work["target_date"], errors="coerce")
    eligible = eligible_mask(work, lookback_weeks)
    assignment_frames = []
    metadata = []
    for fold in folds:
        train_end = pd.Timestamp(fold.train_end)
        validation_start = pd.Timestamp(fold.validation_start)
        validation_end = pd.Timestamp(fold.validation_end)
        train = work.loc[
            eligible & work["trade_date"].le(train_end) & work["target_date"].le(train_end)
        ].copy()
        validation = work.loc[
            eligible
            & work["trade_date"].between(validation_start, validation_end)
            & work["target_date"].le(validation_end)
        ].copy()
        if train.empty or validation.empty:
            raise ValueError(f"{fold.fold_id} has an empty train or validation set")
        if validation["stock_code"].nunique() < minimum_stock_count:
            raise ValueError(f"{fold.fold_id} validation stock coverage below minimum")
        if train["target_date"].max() >= validation["trade_date"].min():
            raise ValueError(f"{fold.fold_id} train targets overlap validation observations")
        for split_name, frame in (("train", train), ("validation", validation)):
            selected = frame[["stock_code", "trade_date", "target_date", "target_return"]].copy()
            selected.insert(0, "split", split_name)
            selected.insert(0, "fold_id", fold.fold_id)
            selected["sample_row_id"] = (
                selected["stock_code"].astype(str)
                + "|" + selected["trade_date"].dt.strftime("%Y-%m-%d")
                + "|" + selected["target_date"].dt.strftime("%Y-%m-%d")
            )
            assignment_frames.append(selected)
        metadata.append({
            **asdict(fold),
            "train_start_effective": train["trade_date"].min().date().isoformat(),
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation)),
            "train_stock_count": int(train["stock_code"].nunique()),
            "validation_stock_count": int(validation["stock_code"].nunique()),
            "train_row_set_sha256": canonical_row_set_sha256(train),
            "validation_row_set_sha256": canonical_row_set_sha256(validation),
            "train_sample_content_sha256": canonical_sample_content_sha256(train),
            "validation_sample_content_sha256": canonical_sample_content_sha256(validation),
        })
    return pd.concat(assignment_frames, ignore_index=True), metadata
