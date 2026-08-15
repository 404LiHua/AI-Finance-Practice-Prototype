from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


class DataCustodyViolation(RuntimeError):
    """Raised before sealed or future data can be opened or processed."""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class DataCustodyGuard:
    repo_root: Path
    policy_id: str
    forbidden_roots: tuple[Path, ...]
    forbidden_name_fragments: tuple[str, ...]
    sealed_start: pd.Timestamp
    sealed_end: pd.Timestamp
    development_date_ceiling: pd.Timestamp

    @classmethod
    def from_config(cls, config_path: Path, repo_root: Path) -> "DataCustodyGuard":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        resolved_root = repo_root.resolve()
        forbidden_roots = tuple(
            (resolved_root / value).resolve()
            for value in config["repo_root_relative_forbidden_paths"]
        )
        return cls(
            repo_root=resolved_root,
            policy_id=str(config["policy_id"]),
            forbidden_roots=forbidden_roots,
            forbidden_name_fragments=tuple(
                str(value).casefold() for value in config["forbidden_name_fragments"]
            ),
            sealed_start=pd.Timestamp(config["sealed_interval"]["start"]),
            sealed_end=pd.Timestamp(config["sealed_interval"]["end"]),
            development_date_ceiling=pd.Timestamp(config["development_date_ceiling"]),
        )

    def assert_path_allowed(self, path: Path, purpose: str = "development") -> Path:
        resolved = path.resolve()
        lowered = resolved.as_posix().casefold()
        for root in self.forbidden_roots:
            if resolved == root or _is_relative_to(resolved, root):
                raise DataCustodyViolation(
                    f"{self.policy_id}: blocked {purpose} path under sealed root: {resolved}"
                )
        for fragment in self.forbidden_name_fragments:
            if fragment in lowered:
                raise DataCustodyViolation(
                    f"{self.policy_id}: blocked {purpose} path containing sealed identifier: {resolved}"
                )
        return resolved

    def assert_paths_allowed(self, paths: Iterable[Path], purpose: str = "development") -> None:
        for path in paths:
            self.assert_path_allowed(path, purpose=purpose)

    def assert_development_frame(
        self,
        frame: pd.DataFrame,
        date_columns: tuple[str, ...] = ("trade_date", "target_date"),
    ) -> None:
        checked = 0
        for column in date_columns:
            if column not in frame.columns:
                continue
            checked += 1
            dates = pd.to_datetime(frame[column], errors="coerce").dropna()
            if dates.empty:
                continue
            sealed = dates.between(self.sealed_start, self.sealed_end)
            if sealed.any():
                first = dates.loc[sealed].min().date().isoformat()
                raise DataCustodyViolation(
                    f"{self.policy_id}: {column} overlaps sealed C-4 interval at {first}"
                )
            beyond = dates.gt(self.development_date_ceiling)
            if beyond.any():
                first = dates.loc[beyond].min().date().isoformat()
                raise DataCustodyViolation(
                    f"{self.policy_id}: {column} exceeds development ceiling at {first}"
                )
        if checked == 0:
            raise DataCustodyViolation(
                f"{self.policy_id}: no auditable date column found in development frame"
            )
