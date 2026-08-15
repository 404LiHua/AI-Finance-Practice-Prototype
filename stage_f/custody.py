from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


class StageFDataCustodyViolation(RuntimeError):
    """Raised before Stage F can read sealed, screening, final, or post-ceiling data."""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class SealedInterval:
    interval_id: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class StageFDataCustodyGuard:
    repo_root: Path
    policy_id: str
    forbidden_roots: tuple[Path, ...]
    forbidden_name_fragments: tuple[str, ...]
    sealed_intervals: tuple[SealedInterval, ...]
    development_date_ceiling: pd.Timestamp

    @classmethod
    def from_config(cls, config_path: Path, repo_root: Path) -> "StageFDataCustodyGuard":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        resolved_root = repo_root.resolve()
        roots = []
        for value in config["forbidden_paths"]:
            path = Path(value)
            roots.append(path.resolve() if path.is_absolute() else (resolved_root / path).resolve())
        intervals = tuple(
            SealedInterval(str(item["interval_id"]), pd.Timestamp(item["start"]), pd.Timestamp(item["end"]))
            for item in config["sealed_intervals"]
        )
        return cls(
            repo_root=resolved_root,
            policy_id=str(config["policy_id"]),
            forbidden_roots=tuple(roots),
            forbidden_name_fragments=tuple(str(value).casefold() for value in config["forbidden_name_fragments"]),
            sealed_intervals=intervals,
            development_date_ceiling=pd.Timestamp(config["development_date_ceiling"]),
        )

    def assert_path_allowed(self, path: Path, purpose: str = "stage_f_development") -> Path:
        resolved = path.resolve()
        lowered = resolved.as_posix().casefold()
        for root in self.forbidden_roots:
            if resolved == root or _is_relative_to(resolved, root):
                raise StageFDataCustodyViolation(f"{self.policy_id}: blocked {purpose} path: {resolved}")
        for fragment in self.forbidden_name_fragments:
            if fragment in lowered:
                raise StageFDataCustodyViolation(
                    f"{self.policy_id}: blocked {purpose} path containing sealed identifier: {resolved}"
                )
        return resolved

    def assert_paths_allowed(self, paths: Iterable[Path], purpose: str = "stage_f_development") -> None:
        for path in paths:
            self.assert_path_allowed(path, purpose)

    def assert_development_dates(self, values: Iterable[object], label: str) -> None:
        dates = pd.to_datetime(pd.Series(list(values)), errors="coerce").dropna()
        if dates.empty:
            raise StageFDataCustodyViolation(f"{self.policy_id}: no auditable dates for {label}")
        beyond = dates.gt(self.development_date_ceiling)
        if beyond.any():
            raise StageFDataCustodyViolation(
                f"{self.policy_id}: {label} exceeds development ceiling at {dates.loc[beyond].min().date()}"
            )
        for interval in self.sealed_intervals:
            overlap = dates.between(interval.start, interval.end)
            if overlap.any():
                raise StageFDataCustodyViolation(
                    f"{self.policy_id}: {label} overlaps {interval.interval_id} at {dates.loc[overlap].min().date()}"
                )
