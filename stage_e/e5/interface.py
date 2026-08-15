from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class E5FoldView:
    fold_id: str
    numeric_values: np.ndarray
    target_raw: np.ndarray
    target_scaled: np.ndarray
    sample_mask: np.ndarray
    node_available: np.ndarray
    split: np.ndarray
    trade_date: np.ndarray
    target_date: np.ndarray
    sample_row_id: np.ndarray
    stock_code: np.ndarray
    text_features: np.ndarray
    text_available: np.ndarray
    text_count: np.ndarray
    target_mean_train: float
    target_std_train: float

    def split_indices(self, split_name: str) -> np.ndarray:
        return np.flatnonzero(self.split.astype(str) == split_name)


class E5ModelAdapter(ABC):
    """Required interface for every Stage E E-5 model family."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build(self, config: dict[str, Any], fold_view: E5FoldView) -> Any:
        raise NotImplementedError

    @abstractmethod
    def fit(self, model: Any, fold_view: E5FoldView, seed: int, artifact_dir: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: Any, fold_view: E5FoldView, split_name: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint: Path, config: dict[str, Any], fold_view: E5FoldView) -> Any:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        raise NotImplementedError


def load_fold_view(adapter_root: Path, fold_id: str, text_view: str) -> E5FoldView:
    fold_root = adapter_root / fold_id
    base_path = fold_root / "base_windows.npz"
    text_path = fold_root / f"text_{text_view}.npz"
    if not base_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(f"missing E-5 fold input: {fold_id} {text_view}")
    base = np.load(base_path)
    text = np.load(text_path)
    base_ids = base["sample_row_id"].astype(str)
    text_ids = text["sample_row_id"].astype(str)
    if not np.array_equal(base_ids, text_ids):
        raise ValueError(f"text view changes frozen sample keys: {fold_id} {text_view}")
    if base["target_raw"].shape != base_ids.shape or text["text_available"].shape != base_ids.shape:
        raise ValueError("E-5 fold arrays do not share the frozen panel shape")
    return E5FoldView(
        fold_id=fold_id, numeric_values=base["values"].astype(np.float32),
        target_raw=base["target_raw"].astype(np.float32), target_scaled=base["target_scaled"].astype(np.float32),
        sample_mask=base["sample_mask"].astype(bool), node_available=base["node_available"].astype(bool),
        split=base["split"].astype(str), trade_date=base["trade_date"].astype(str),
        target_date=base["target_date"].astype(str), sample_row_id=base_ids,
        stock_code=base["stock_code"].astype(str), text_features=text["features"].astype(np.float32),
        text_available=text["text_available"].astype(bool), text_count=text["text_count"].astype(np.int64),
        target_mean_train=float(base["target_mean_train"][0]), target_std_train=float(base["target_std_train"][0]),
    )


def validation_key_frame(view: E5FoldView) -> pd.DataFrame:
    rows = []
    for index in view.split_indices("validation"):
        for stock_index, stock in enumerate(view.stock_code):
            rows.append({
                "fold_id": view.fold_id, "sample_row_id": str(view.sample_row_id[index, stock_index]),
                "trade_date": str(view.trade_date[index]), "target_date": str(view.target_date[index, stock_index]),
                "stock_code": str(stock), "target_return": float(view.target_raw[index, stock_index]),
                "sample_valid": bool(view.sample_mask[index, stock_index]),
                "text_available": bool(view.text_available[index, stock_index]),
            })
    return pd.DataFrame(rows)
