from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


class StageEDataCustodyViolation(RuntimeError):
    """Raised before sealed, screening, final, or post-ceiling data can be read."""


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
class StageEDataCustodyGuard:
    repo_root: Path
    policy_id: str
    forbidden_roots: tuple[Path, ...]
    forbidden_name_fragments: tuple[str, ...]
    sealed_intervals: tuple[SealedInterval, ...]
    development_date_ceiling: pd.Timestamp

    @classmethod
    def from_config(cls, config_path: Path, repo_root: Path) -> "StageEDataCustodyGuard":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        resolved_root = repo_root.resolve()
        forbidden_roots = []
        for value in config["forbidden_paths"]:
            path = Path(value)
            forbidden_roots.append(path.resolve() if path.is_absolute() else (resolved_root / path).resolve())
        intervals = tuple(
            SealedInterval(
                interval_id=str(item["interval_id"]),
                start=pd.Timestamp(item["start"]),
                end=pd.Timestamp(item["end"]),
            )
            for item in config["sealed_intervals"]
        )
        return cls(
            repo_root=resolved_root,
            policy_id=str(config["policy_id"]),
            forbidden_roots=tuple(forbidden_roots),
            forbidden_name_fragments=tuple(
                str(value).casefold() for value in config["forbidden_name_fragments"]
            ),
            sealed_intervals=intervals,
            development_date_ceiling=pd.Timestamp(config["development_date_ceiling"]),
        )

    def assert_path_allowed(self, path: Path, purpose: str = "stage_e_development") -> Path:
        resolved = path.resolve()
        lowered = resolved.as_posix().casefold()
        for root in self.forbidden_roots:
            if resolved == root or _is_relative_to(resolved, root):
                raise StageEDataCustodyViolation(
                    f"{self.policy_id}: blocked {purpose} path under sealed root: {resolved}"
                )
        for fragment in self.forbidden_name_fragments:
            if fragment in lowered:
                raise StageEDataCustodyViolation(
                    f"{self.policy_id}: blocked {purpose} path containing sealed identifier: {resolved}"
                )
        return resolved

    def assert_paths_allowed(self, paths: Iterable[Path], purpose: str = "stage_e_development") -> None:
        for path in paths:
            self.assert_path_allowed(path, purpose=purpose)

    def assert_development_frame(
        self,
        frame: pd.DataFrame,
        date_columns: tuple[str, ...] = ("trade_date", "target_date"),
    ) -> None:
        checked = 0
        for column in date_columns:
            if column not in frame:
                continue
            checked += 1
            dates = pd.to_datetime(frame[column], errors="coerce").dropna()
            if dates.empty:
                continue
            beyond = dates.gt(self.development_date_ceiling)
            if beyond.any():
                first = dates.loc[beyond].min().date().isoformat()
                raise StageEDataCustodyViolation(
                    f"{self.policy_id}: {column} exceeds development ceiling at {first}"
                )
            for interval in self.sealed_intervals:
                overlap = dates.between(interval.start, interval.end)
                if overlap.any():
                    first = dates.loc[overlap].min().date().isoformat()
                    raise StageEDataCustodyViolation(
                        f"{self.policy_id}: {column} overlaps {interval.interval_id} at {first}"
                    )
        if checked == 0:
            raise StageEDataCustodyViolation(
                f"{self.policy_id}: no auditable date column found in development frame"
            )
