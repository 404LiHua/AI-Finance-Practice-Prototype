from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


NUMERIC_FEATURES = [
    "model_open", "model_high", "model_low", "model_close",
    "model_volume_hands", "model_amount_thousand_cny",
    "return_1w", "log_return_1w", "intraweek_range", "candle_body",
    "close_ma_4", "close_ma_12", "close_ma_26",
    "return_vol_4", "return_vol_12", "return_vol_26",
    "log_volume", "volume_z_12", "weeks_since_listing",
    "csmar_total_shares", "csmar_tradable_a_shares", "text_count",
]


def load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_path"] = str(path.resolve())
    for key in ("data_root", "output_root"):
        value = Path(config[key])
        if not value.is_absolute():
            value = repo_root / value
        config[key] = str(value.resolve())
    return config


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    result: dict[str, Any] = {"python": seed, "numpy": seed, "torch": None}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        result["torch"] = seed
    except ImportError:
        pass
    return result


def create_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"stage_b.{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def environment_info(repo_root: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "sklearn", "statsmodels", "torch"):
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[package] = None
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root}", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "git_commit": git_commit,
    }


@dataclass
class DataBundle:
    panel: pd.DataFrame
    samples: dict[str, pd.DataFrame]
    feature_columns: list[str]

    @classmethod
    def load(cls, data_root: Path) -> "DataBundle":
        panel = pd.read_csv(data_root / "panel.csv.gz", low_memory=False)
        text_path = data_root / "text_features.csv.gz"
        if text_path.exists():
            text = pd.read_csv(text_path)
            keep = [
                c for c in text.columns
                if c in {"stock_code", "calendar_week_end", "text_cluster"}
                or c.startswith("text_svd_")
            ]
            panel = panel.merge(
                text[keep].drop_duplicates(["stock_code", "calendar_week_end"]),
                on=["stock_code", "calendar_week_end"], how="left", validate="one_to_one",
            )
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel["target_date"] = pd.to_datetime(panel["target_date"], errors="coerce")
        panel = panel.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
        feature_columns = [c for c in NUMERIC_FEATURES if c in panel.columns]
        feature_columns += sorted(c for c in panel.columns if c.startswith("text_svd_"))
        samples: dict[str, pd.DataFrame] = {}
        for split in ("train", "validation", "test"):
            frame = panel[(panel["split"] == split) & panel["sample_eligible"].fillna(False)].copy()
            samples[split] = frame.reset_index(drop=True)
        return cls(panel=panel, samples=samples, feature_columns=feature_columns)


def evaluate_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["target_return"].to_numpy(dtype=float)
    y_pred = frame["prediction"].to_numpy(dtype=float)
    error = y_pred - y_true
    nonzero = np.abs(y_true) > 1e-8
    true_direction = (y_true > 0).astype(int)
    pred_direction = (y_pred > 0).astype(int)
    aggregate = {
        "samples": int(len(frame)),
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape_pct": float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else None,
        "direction_accuracy": float(np.mean(true_direction == pred_direction)),
        "direction_f1": float(f1_score(true_direction, pred_direction, zero_division=0)),
    }
    per_stock: dict[str, Any] = {}
    for stock_code, stock_frame in frame.groupby("stock_code", sort=True):
        stock_true = stock_frame["target_return"].to_numpy(dtype=float)
        stock_pred = stock_frame["prediction"].to_numpy(dtype=float)
        stock_error = stock_pred - stock_true
        per_stock[str(stock_code)] = {
            "samples": int(len(stock_frame)),
            "mae": float(np.mean(np.abs(stock_error))),
            "rmse": float(np.sqrt(np.mean(stock_error ** 2))),
            "direction_accuracy": float(np.mean((stock_true > 0) == (stock_pred > 0))),
        }
    return {"aggregate": aggregate, "per_stock": per_stock}


def prediction_frame(samples: pd.DataFrame, prediction: np.ndarray, split: str) -> pd.DataFrame:
    result = samples[[
        "stock_code", "trade_date", "target_date", "model_close",
        "target_close", "target_return", "target_direction",
    ]].copy()
    result.insert(0, "split", split)
    result["prediction"] = np.asarray(prediction, dtype=float)
    result["predicted_close"] = result["model_close"] * (1.0 + result["prediction"])
    result["absolute_error"] = (result["prediction"] - result["target_return"]).abs()
    return result


class Timer:
    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.seconds = time.perf_counter() - self.started
