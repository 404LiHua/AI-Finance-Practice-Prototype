"""Build frozen-key E-4 fold tensors from E-3 numeric windows and E-2 text views."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.array_io import write_deterministic_npz
from stage_e.custody import StageEDataCustodyGuard
from stage_e.hashing import sha256_file, stable_json_sha256

DEFAULT_CUSTODY = REPO_ROOT / "stage_e/configs/data_custody_v1.json"
KEY_COLUMNS = ["fold_id", "split", "stock_code", "trade_date", "target_date", "sample_row_id"]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_view(path: Path, fold_id: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if fold_id is not None:
        frame = frame[frame["fold_id"].astype(str).eq(fold_id)].copy()
    return frame


def assert_key_equality(assignments: pd.DataFrame, view: pd.DataFrame, view_name: str) -> None:
    expected = assignments[KEY_COLUMNS].astype(str).sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    actual = view[KEY_COLUMNS].astype(str).sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    if not expected.equals(actual):
        raise ValueError(f"{view_name} keys differ from frozen fold assignments")


def align_text(
    view: pd.DataFrame,
    feature_columns: list[str],
    sample_positions: dict[str, tuple[int, int]],
    shape: tuple[int, int],
    sample_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    features = np.zeros((*shape, len(feature_columns)), dtype=np.float32)
    available = np.zeros(shape, dtype=np.bool_)
    count = np.zeros(shape, dtype=np.int32)
    for row in view.itertuples(index=False):
        position = sample_positions[str(row.sample_row_id)]
        available[position] = bool(row.text_available)
        count[position] = int(row.text_count)
        if feature_columns:
            values = np.asarray([getattr(row, column) for column in feature_columns], dtype=np.float32)
            values[~np.isfinite(values)] = 0.0
            features[position] = values
    features[~available] = 0.0
    return {
        "features": features, "text_available": available, "text_count": count,
        "sample_row_id": sample_ids.copy(), "sample_mask": sample_ids != "",
    }


def build(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    guard = StageEDataCustodyGuard.from_config(DEFAULT_CUSTODY, REPO_ROOT)
    panel_path = guard.assert_path_allowed(resolve(config["paths"]["panel"]), purpose="E-4 panel")
    order_path = guard.assert_path_allowed(resolve(config["paths"]["stock_order"]), purpose="E-4 stock order")
    assignments_path = guard.assert_path_allowed(resolve(config["paths"]["fold_assignments"]), purpose="E-4 assignments")
    e2_root = guard.assert_path_allowed(resolve(config["paths"]["e2_root"]), purpose="E-4 E-2 views")
    output_root = guard.assert_path_allowed(resolve(config["paths"]["output_root"]), purpose="E-4 adapter output")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    features = list(config["feature_columns"])
    usecols = [
        "trade_date", "stock_code", "is_market_open_week", "model_eligible_pit",
        config["target_column"], *features,
    ]
    panel = pd.read_csv(panel_path, usecols=usecols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    guard.assert_development_frame(panel, date_columns=("trade_date",))
    if panel["trade_date"].max() > pd.Timestamp(config["development_date_ceiling"]):
        raise ValueError("E-4 panel exceeds development ceiling")
    order = pd.read_csv(order_path).sort_values("selection_rank", kind="stable")
    stocks = order["stock_code"].astype(str).tolist()
    stock_index = {stock: index for index, stock in enumerate(stocks)}
    dates = pd.DatetimeIndex(sorted(panel.loc[panel["is_market_open_week"].astype(bool), "trade_date"].dropna().unique()))
    date_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    dense_index = pd.MultiIndex.from_product([dates, stocks], names=["trade_date", "stock_code"])
    dense = panel.set_index(["trade_date", "stock_code"]).reindex(dense_index)
    feature_arrays = []
    for column in features:
        values = pd.to_numeric(dense[column], errors="coerce").to_numpy(dtype=np.float64).reshape(len(dates), len(stocks))
        if column == "model_close":
            values = np.log(np.clip(values, 1e-8, None))
        elif column == "model_volume_hands":
            values = np.log1p(np.clip(values, 0.0, None))
        feature_arrays.append(values)
    feature_cube = np.stack(feature_arrays, axis=-1)
    target_cube = pd.to_numeric(dense[config["target_column"]], errors="coerce").to_numpy(dtype=np.float64).reshape(len(dates), len(stocks))
    available_cube = dense["model_eligible_pit"].fillna(False).astype(bool).to_numpy().reshape(len(dates), len(stocks))
    assignments = pd.read_csv(assignments_path)
    assignments["trade_date"] = pd.to_datetime(assignments["trade_date"], errors="coerce")
    assignments["target_date"] = pd.to_datetime(assignments["target_date"], errors="coerce")
    guard.assert_development_frame(assignments, date_columns=("trade_date", "target_date"))
    no_text_path = e2_root / config["text_views"]["no_text"]
    semantic_path = e2_root / config["text_views"]["semantic"]
    no_text_all = read_view(no_text_path)
    semantic_all = read_view(semantic_path)
    e2_metadata_path = e2_root / "metadata.json"
    e2_metadata = json.loads(e2_metadata_path.read_text(encoding="utf-8"))
    tfidf_metadata = {item["fold_id"]: item for item in e2_metadata["artifacts"]["tfidf_svd"]}
    fold_receipts = []
    lookback = int(config["lookback_weeks"])
    for fold_id in sorted(assignments["fold_id"].astype(str).unique()):
        fold = assignments[assignments["fold_id"].astype(str).eq(fold_id)].copy()
        fold_dates = pd.DatetimeIndex(sorted(fold["trade_date"].unique()))
        split_by_date = fold.groupby("trade_date")["split"].first().reindex(fold_dates).astype(str)
        windows = []
        current_available = []
        for date in fold_dates:
            current = date_index[pd.Timestamp(date)]
            if current < lookback - 1:
                raise ValueError(f"{fold_id} date lacks {lookback} weeks of history: {date}")
            windows.append(feature_cube[current - lookback + 1 : current + 1])
            current_available.append(available_cube[current])
        raw_windows = np.stack(windows)
        current_available_array = np.stack(current_available)
        train_dates = split_by_date.eq("train").to_numpy()
        train_values = raw_windows[train_dates]
        means = np.asarray([np.nanmean(train_values[..., index]) for index in range(len(features))])
        stds = np.asarray([np.nanstd(train_values[..., index]) for index in range(len(features))])
        stds = np.where((~np.isfinite(stds)) | (stds < 1e-8), 1.0, stds)
        normalized = (raw_windows - means) / stds
        normalized[~np.isfinite(normalized)] = 0.0
        sample_ids = np.full((len(fold_dates), len(stocks)), "", dtype="<U64")
        target_dates = np.full((len(fold_dates), len(stocks)), "", dtype="<U10")
        target_raw = np.full((len(fold_dates), len(stocks)), np.nan, dtype=np.float32)
        date_position = {pd.Timestamp(date): index for index, date in enumerate(fold_dates)}
        sample_positions: dict[str, tuple[int, int]] = {}
        for row in fold.itertuples(index=False):
            position = (date_position[pd.Timestamp(row.trade_date)], stock_index[str(row.stock_code)])
            sample_id = str(row.sample_row_id)
            sample_positions[sample_id] = position
            sample_ids[position] = sample_id
            target_dates[position] = pd.Timestamp(row.target_date).strftime("%Y-%m-%d")
            target_raw[position] = float(row.target_return)
        sample_mask = sample_ids != ""
        train_sample_mask = sample_mask & train_dates[:, None]
        train_targets = target_raw[train_sample_mask]
        target_mean = float(np.mean(train_targets))
        target_std = float(np.std(train_targets))
        if target_std < 1e-8:
            raise ValueError(f"{fold_id} target standard deviation is too small")
        target_scaled = (target_raw - target_mean) / target_std
        target_scaled[~sample_mask] = 0.0
        fold_root = output_root / fold_id
        fold_root.mkdir(parents=True, exist_ok=True)
        base_path = fold_root / "base_windows.npz"
        write_deterministic_npz(base_path, {
            "values": normalized.astype(np.float32), "target_raw": target_raw,
            "target_scaled": target_scaled.astype(np.float32), "sample_mask": sample_mask,
            "node_available": current_available_array.astype(np.bool_), "sample_row_id": sample_ids,
            "target_date": target_dates, "trade_date": np.asarray([date.strftime("%Y-%m-%d") for date in fold_dates], dtype="<U10"),
            "split": split_by_date.to_numpy(dtype="<U10"), "stock_code": np.asarray(stocks, dtype="<U16"),
            "feature_name": np.asarray(features, dtype="<U32"), "feature_mean_train": means.astype(np.float64),
            "feature_std_train": stds.astype(np.float64), "target_mean_train": np.asarray([target_mean]),
            "target_std_train": np.asarray([target_std]),
        })
        fold_no_text = no_text_all[no_text_all["fold_id"].astype(str).eq(fold_id)].copy()
        fold_semantic = semantic_all[semantic_all["fold_id"].astype(str).eq(fold_id)].copy()
        tfidf_path = e2_root / config["text_views"]["tfidf_svd_pattern"].format(fold_id=fold_id)
        tfidf_model_path = e2_root / config["text_views"]["tfidf_model_pattern"].format(fold_id=fold_id)
        fold_tfidf = read_view(tfidf_path)
        assert_key_equality(fold, fold_no_text, f"{fold_id} no_text")
        assert_key_equality(fold, fold_tfidf, f"{fold_id} tfidf_svd")
        assert_key_equality(fold, fold_semantic, f"{fold_id} semantic")
        views = {
            "no_text": (fold_no_text, []),
            "tfidf_svd": (fold_tfidf, [column for column in fold_tfidf.columns if column.startswith("text_svd_")]),
            "semantic_bge": (fold_semantic, [column for column in fold_semantic.columns if column.startswith("text_semantic_")]),
        }
        view_receipts = {}
        for view_name, (view_frame, columns) in views.items():
            view_path = fold_root / f"text_{view_name}.npz"
            arrays = align_text(view_frame, columns, sample_positions, sample_ids.shape, sample_ids)
            arrays["feature_name"] = np.asarray(columns, dtype="<U32")
            write_deterministic_npz(view_path, arrays)
            view_receipts[view_name] = {
                "dimension": len(columns), "text_available_rows": int(arrays["text_available"].sum()),
                "missing_text_rows": int(sample_mask.sum() - arrays["text_available"].sum()),
                "artifact_sha256": sha256_file(view_path),
            }
        receipt = {
            "fold_id": fold_id, "cross_section_count": len(fold_dates), "stock_count": len(stocks),
            "assignment_rows": int(len(fold)), "sample_mask_rows": int(sample_mask.sum()),
            "train_cross_sections": int(train_dates.sum()), "validation_cross_sections": int((~train_dates).sum()),
            "base_windows_sha256": sha256_file(base_path), "views": view_receipts,
            "tfidf_train_fitted_model_sha256": sha256_file(tfidf_model_path),
            "tfidf_upstream_declared_model_sha256": tfidf_metadata[fold_id]["model_sha256"],
            "tfidf_training_documents": tfidf_metadata[fold_id]["training_documents"],
            "tfidf_validation_documents": tfidf_metadata[fold_id]["validation_documents"],
        }
        if receipt["tfidf_train_fitted_model_sha256"] != receipt["tfidf_upstream_declared_model_sha256"]:
            raise ValueError(f"{fold_id} TF-IDF/SVD fitted model hash mismatch")
        fold_receipts.append(receipt)
    metadata = {
        "stage": "E-4.2", "adapter_batch_id": config["adapter_batch_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_date_ceiling": config["development_date_ceiling"], "lookback_weeks": lookback,
        "stock_count": len(stocks), "feature_columns": features, "folds": fold_receipts,
        "text_join": "left join by frozen sample_row_id; missing text retained as zero vector",
        "tfidf_fit_policy": "no refit in E-4; use E-2 per-fold TRAIN-fitted model artifacts only",
        "config_sha256": sha256_file(config_path), "panel_sha256": sha256_file(panel_path),
        "stock_order_sha256": sha256_file(order_path), "assignments_sha256": sha256_file(assignments_path),
        "e2_metadata_sha256": sha256_file(e2_metadata_path), "future_or_sealed_data_read": False,
    }
    metadata["batch_sha256"] = stable_json_sha256(metadata)
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(build(config_path, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
