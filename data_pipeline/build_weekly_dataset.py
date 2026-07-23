"""Build a traceable, leakage-aware weekly A-share research dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_COLUMNS = {
    "股票代码": "stock_code",
    "交易日期": "trade_date",
    "周收盘价": "close",
    "周开盘价": "open",
    "周最高价": "high",
    "周最低价": "low",
    "上周收盘价": "previous_close",
    "周涨跌额": "price_change",
    "周涨跌幅(%)": "return_reported",
    "周成交量(手)": "volume_hands",
    "周成交额(千元)": "amount_thousand_cny",
}
NUMERIC_COLUMNS = list(RAW_COLUMNS.values())[2:]
BASIC_COLUMNS = {
    "股票代码": "stock_code",
    "股票名称": "stock_name",
    "地区": "region",
    "行业": "industry",
    "公司全称": "company_name",
    "英文名称": "english_name",
    "拼音缩写": "pinyin_abbreviation",
    "市场类型": "market_type",
    "交易所代码": "exchange_code",
    "交易货币": "currency",
    "上市状态": "listing_status_current",
    "上市日期": "listing_date",
    "是否沪深港通标的": "stock_connect_status_current",
    "实控人名称": "controller_name_current",
    "企业性质": "enterprise_nature_current",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    split = config["split"]
    total = split["train_ratio"] + split["validation_ratio"] + split["test_ratio"]
    if not np.isclose(total, 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    if config.get("strict_provenance"):
        required = ["provider", "retrieval_method", "license_or_terms", "price_adjustment"]
        missing = [f"source.{key}" for key in required if "USER_CONFIRM_REQUIRED" in str(config["source"].get(key, ""))]
        if config.get("stock_basic", {}).get("enabled"):
            basic_required = ["provider", "retrieval_method", "license_or_terms"]
            missing.extend(
                f"stock_basic.{key}" for key in basic_required
                if "USER_CONFIRM_REQUIRED" in str(config["stock_basic"].get(key, ""))
            )
        if missing:
            raise ValueError(f"strict provenance requires confirmed fields: {missing}")
    return config


def discover_files(source_root: Path) -> list[Path]:
    files = sorted(source_root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found in {source_root}")
    return files


def inspect_file(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=["股票代码", "交易日期"])
    dates = pd.to_datetime(frame["交易日期"], errors="coerce")
    return {
        "file_name": path.name,
        "stock_code": path.stem,
        "size_bytes": path.stat().st_size,
        "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
        "row_count": int(len(frame)),
        "date_min": None if dates.isna().all() else dates.min().date().isoformat(),
        "date_max": None if dates.isna().all() else dates.max().date().isoformat(),
        "bad_date_count": int(dates.isna().sum()),
        "code_mismatch_count": int((frame["股票代码"].astype(str) != path.stem).sum()),
    }


def choose_universe(manifest: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    options = config["universe"]
    explicit = [str(code) for code in options.get("codes", [])]
    if explicit:
        absent = sorted(set(explicit) - set(manifest["stock_code"]))
        if absent:
            raise ValueError(f"explicit stock codes not found: {absent[:20]}")
        return explicit
    eligible = manifest.loc[manifest["row_count"] >= int(options["min_weeks"])].copy()
    eligible = eligible.sort_values(["row_count", "stock_code"], ascending=[False, True])
    if options.get("max_stocks"):
        eligible = eligible.head(int(options["max_stocks"]))
    if eligible.empty:
        raise ValueError("no stocks satisfy the universe rules")
    return eligible["stock_code"].tolist()


def load_price_file(path: Path, file_hash: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    frame = frame.rename(columns=RAW_COLUMNS)[list(RAW_COLUMNS.values())]
    frame["source_file"] = path.name
    frame["source_sha256"] = file_hash
    frame["source_row_number"] = np.arange(2, len(frame) + 2, dtype=np.int64)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def apply_date_range(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    options = config.get("date_range", {})
    start = pd.Timestamp(options["start"]) if options.get("start") else None
    end = pd.Timestamp(options["end"]) if options.get("end") else None
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame["trade_date"].ge(start)
    if end is not None:
        mask &= frame["trade_date"].le(end)
    result = frame.loc[mask].copy()
    if result.empty:
        raise ValueError(f"date_range produced no rows: {options}")
    return result


def load_stock_basic(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    options = config.get("stock_basic", {})
    if not options.get("enabled"):
        return pd.DataFrame(columns=list(BASIC_COLUMNS.values())), {"enabled": False}
    path = Path(options["path"])
    if not path.exists():
        raise FileNotFoundError(f"stock basic input not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(BASIC_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"stock basic file missing columns: {missing}")
    frame = frame.rename(columns=BASIC_COLUMNS)[list(BASIC_COLUMNS.values())]
    if frame["stock_code"].duplicated().any():
        raise ValueError("stock basic contains duplicate stock_code values")
    frame["listing_date"] = pd.to_datetime(frame["listing_date"], errors="coerce")
    frame["stock_basic_source_sha256"] = sha256_file(path)
    manifest = {
        "enabled": True,
        "file_name": path.name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": frame["stock_basic_source_sha256"].iloc[0],
        "row_count": int(len(frame)),
        "duplicate_stock_code": int(frame["stock_code"].duplicated().sum()),
        "bad_listing_date": int(frame["listing_date"].isna().sum()),
        "provider": options.get("provider"),
        "retrieval_method": options.get("retrieval_method"),
        "retrieved_at": options.get("retrieved_at"),
        "license_or_terms": options.get("license_or_terms"),
    }
    return frame, manifest


def merge_stock_basic(prices: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    if stock_basic.empty:
        prices = prices.copy()
        prices["stock_basic_available"] = False
        return prices
    merged = prices.merge(stock_basic, on="stock_code", how="left", validate="many_to_one")
    merged["stock_basic_available"] = merged["stock_name"].notna()
    merged["before_listing_date"] = merged["listing_date"].notna() & merged["trade_date"].lt(merged["listing_date"])
    merged["weeks_since_listing"] = (
        (merged["trade_date"] - merged["listing_date"]).dt.days.div(7).where(~merged["before_listing_date"])
    )
    return merged


def load_baostock_weekly(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    options = config.get("baostock", {})
    if not options.get("enabled"):
        return pd.DataFrame(), {"enabled": False}
    root = Path(options["root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"BaoStock manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {item["file"]: item["sha256"] for item in manifest["files"]}
    frames = []
    for path in sorted(root.glob("*.weekly_qfq.csv")):
        frame = pd.read_csv(path)
        frame = frame.rename(columns={
            "project_stock_code": "stock_code",
            "date": "trade_date",
            "open": "qfq_open",
            "high": "qfq_high",
            "low": "qfq_low",
            "close": "qfq_close",
            "volume": "qfq_volume",
            "amount": "qfq_amount",
            "adjustflag": "baostock_adjustflag",
            "turn": "baostock_turnover_rate",
            "pctChg": "baostock_return_pct",
        })
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        for column in (
            "qfq_open", "qfq_high", "qfq_low", "qfq_close", "qfq_volume",
            "qfq_amount", "baostock_turnover_rate", "baostock_return_pct",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["baostock_source_file"] = path.name
        frame["baostock_source_sha256"] = hashes.get(path.name, sha256_file(path))
        frames.append(frame[[
            "stock_code", "trade_date", "qfq_open", "qfq_high", "qfq_low", "qfq_close",
            "qfq_volume", "qfq_amount", "baostock_adjustflag", "baostock_turnover_rate",
            "baostock_return_pct", "baostock_source_file", "baostock_source_sha256",
        ]])
    if not frames:
        raise FileNotFoundError(f"no BaoStock weekly files found in {root}")
    return pd.concat(frames, ignore_index=True), manifest


def merge_adjusted_prices(
    prices: pd.DataFrame, adjusted: pd.DataFrame, required: bool
) -> pd.DataFrame:
    if adjusted.empty:
        if required:
            raise ValueError("BaoStock adjusted prices are required but unavailable")
        prices = prices.copy()
        prices["baostock_available"] = False
    else:
        prices = prices.merge(adjusted, on=["stock_code", "trade_date"], how="left", validate="one_to_one")
        prices["baostock_available"] = prices["qfq_close"].notna()
        if required and not prices["baostock_available"].all():
            missing = prices.loc[~prices["baostock_available"], ["stock_code", "trade_date"]]
            raise ValueError(f"BaoStock data missing for {len(missing)} selected rows")
    for raw, adjusted_name in (
        ("open", "qfq_open"), ("high", "qfq_high"), ("low", "qfq_low"),
        ("close", "qfq_close"), ("volume_hands", "qfq_volume"),
        ("amount_thousand_cny", "qfq_amount"),
    ):
        model_name = f"model_{raw}"
        if adjusted_name in prices:
            prices[model_name] = prices[adjusted_name].where(prices["baostock_available"], prices[raw])
        else:
            prices[model_name] = prices[raw]
    prices["model_price_basis"] = np.where(prices["baostock_available"], "BaoStock前复权", "九章量化未复权")
    return prices


def validate_prices(frame: pd.DataFrame) -> dict[str, int]:
    invalid_ohlc = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    ordered = frame.sort_values(["stock_code", "trade_date"])
    observed_previous = ordered.groupby("stock_code", sort=False)["close"].shift(1)
    previous_close_mismatch = (
        observed_previous.notna()
        & ordered["previous_close"].notna()
        & (observed_previous - ordered["previous_close"]).abs().gt(1e-6)
    )
    calculated_return = ordered["price_change"] / ordered["previous_close"].replace(0, np.nan)
    reported_return_mismatch = (
        calculated_return.notna()
        & ordered["return_reported"].notna()
        & (calculated_return - ordered["return_reported"]).abs().gt(5e-4)
    )
    return {
        "rows": int(len(frame)),
        "duplicate_stock_date": int(frame.duplicated(["stock_code", "trade_date"]).sum()),
        "bad_dates": int(frame["trade_date"].isna().sum()),
        "missing_numeric_cells": int(frame[NUMERIC_COLUMNS].isna().sum().sum()),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "nonpositive_price_rows": int((frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "previous_close_continuity_mismatch_rows": int(previous_close_mismatch.sum()),
        "reported_return_mismatch_rows": int(reported_return_mismatch.sum()),
    }


def add_calendar_and_features(frame: pd.DataFrame) -> pd.DataFrame:
    for raw in ("open", "high", "low", "close", "volume_hands", "amount_thousand_cny"):
        model_name = f"model_{raw}"
        if model_name not in frame:
            frame[model_name] = frame[raw]
    frame = frame.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    frame["calendar_week_end"] = frame["trade_date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    frame["is_friday_observation"] = frame["trade_date"].dt.dayofweek.eq(4)
    frame["observation_lag_days"] = (frame["calendar_week_end"] - frame["trade_date"]).dt.days
    grouped = frame.groupby("stock_code", sort=False, group_keys=False)
    frame["return_1w"] = grouped["model_close"].pct_change(fill_method=None)
    frame["log_return_1w"] = grouped["model_close"].transform(lambda values: np.log(values).diff())
    model_previous_close = grouped["model_close"].shift(1)
    frame["intraweek_range"] = (frame["model_high"] - frame["model_low"]) / model_previous_close.replace(0, np.nan)
    frame["candle_body"] = (frame["model_close"] - frame["model_open"]) / frame["model_open"].replace(0, np.nan)
    for window in (4, 12, 26):
        frame[f"close_ma_{window}"] = grouped["model_close"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).mean()
        )
        frame[f"return_vol_{window}"] = grouped["return_1w"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).std()
        )
    frame["log_volume"] = np.log1p(frame["model_volume_hands"].clip(lower=0))
    by_code_log_volume = frame["log_volume"].groupby(frame["stock_code"])
    rolling_mean = by_code_log_volume.transform(lambda values: values.rolling(12, min_periods=12).mean())
    rolling_std = by_code_log_volume.transform(lambda values: values.rolling(12, min_periods=12).std())
    frame["volume_z_12"] = (frame["log_volume"] - rolling_mean) / rolling_std.replace(0, np.nan)
    return frame


def load_text(config: dict[str, Any]) -> pd.DataFrame:
    text_config = config["text"]
    output_columns = [
        "stock_code", "calendar_week_end", "text_count", "text_title",
        "text_body", "text_sources", "text_urls", "text_source_hashes",
        "text_source_rows", "text_available",
    ]
    if not text_config.get("enabled"):
        return pd.DataFrame(columns=output_columns)
    path = Path(text_config["path"])
    if not path.exists():
        raise FileNotFoundError(f"text input not found: {path}")
    text = pd.read_json(path, lines=True) if path.suffix.lower() == ".jsonl" else pd.read_csv(
        path, encoding=text_config.get("encoding", "utf-8")
    )
    mapping = text_config["columns"]
    rename = {source: target for target, source in mapping.items() if source in text.columns}
    text = text.rename(columns=rename)
    missing = sorted({"published_at", "stock_code", "title"} - set(text.columns))
    if missing:
        raise ValueError(f"text data missing normalized columns: {missing}")
    text["published_at"] = pd.to_datetime(text["published_at"], errors="coerce")
    text = text.dropna(subset=["published_at", "stock_code"])
    text["calendar_week_end"] = text["published_at"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    for column in ("title", "body", "source", "url", "source_sha256", "source_row_number"):
        if column not in text:
            text[column] = ""
        text[column] = text[column].fillna("").astype(str)
    aggregated = text.groupby(["stock_code", "calendar_week_end"], as_index=False).agg(
        text_count=("title", "size"),
        text_title=("title", lambda values: " [SEP] ".join(value for value in values if value)),
        text_body=("body", lambda values: " [SEP] ".join(value for value in values if value)),
        text_sources=("source", lambda values: "|".join(sorted(set(value for value in values if value)))),
        text_urls=("url", lambda values: "|".join(value for value in values if value)),
        text_source_hashes=("source_sha256", lambda values: "|".join(sorted(set(value for value in values if value)))),
        text_source_rows=("source_row_number", lambda values: "|".join(value for value in values if value)),
    )
    aggregated["text_available"] = True
    return aggregated[output_columns]


def merge_text(prices: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    merged = prices.merge(text, on=["stock_code", "calendar_week_end"], how="left")
    merged["text_count"] = merged["text_count"].fillna(0).astype(np.int32)
    merged["text_available"] = merged["text_available"].fillna(False).astype(bool)
    for column in (
        "text_title", "text_body", "text_sources", "text_urls",
        "text_source_hashes", "text_source_rows",
    ):
        merged[column] = merged[column].fillna("")
    return merged


def load_csmar_weekly(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    options = config.get("csmar", {})
    if not options.get("enabled"):
        return pd.DataFrame(), {"enabled": False}
    project_root = Path(__file__).resolve().parents[1]
    feature_path = Path(options["weekly_features_path"])
    manifest_path = Path(options["manifest_path"])
    if not feature_path.is_absolute():
        feature_path = project_root / feature_path
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    if not feature_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("prepared CSMAR weekly features or manifest is missing")
    frame = pd.read_csv(feature_path)
    frame["calendar_week_end"] = pd.to_datetime(frame["calendar_week_end"], errors="coerce")
    if frame.duplicated(["stock_code", "calendar_week_end"]).any():
        raise ValueError("CSMAR weekly features contain duplicate stock/week rows")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return frame, manifest


def merge_csmar_weekly(prices: pd.DataFrame, csmar: pd.DataFrame) -> pd.DataFrame:
    if csmar.empty:
        prices = prices.copy()
        prices["csmar_weekly_available"] = False
        return prices
    merged = prices.merge(csmar, on=["stock_code", "calendar_week_end"], how="left", validate="one_to_one")
    merged["csmar_weekly_available"] = merged["csmar_special_status"].notna()
    return merged


def assign_splits(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    options = config["split"]
    dates = pd.Index(sorted(frame["calendar_week_end"].dropna().unique()))
    if len(dates) < 10:
        raise ValueError("not enough distinct weeks for chronological split")
    train_count = max(1, int(len(dates) * options["train_ratio"]))
    validation_count = max(1, int(len(dates) * options["validation_ratio"]))
    purge = int(options.get("purge_weeks", 0))
    validation_start_index = train_count + purge
    validation_end_index = validation_start_index + validation_count - 1
    test_start_index = validation_end_index + 1 + purge
    if test_start_index >= len(dates):
        raise ValueError("purge and split configuration leaves no test data")
    boundaries = {
        "train_end": pd.Timestamp(dates[train_count - 1]),
        "validation_start": pd.Timestamp(dates[validation_start_index]),
        "validation_end": pd.Timestamp(dates[validation_end_index]),
        "test_start": pd.Timestamp(dates[test_start_index]),
        "test_end": pd.Timestamp(dates[-1]),
    }
    frame = frame.copy()
    frame["split"] = "purged"
    frame.loc[frame["calendar_week_end"] <= boundaries["train_end"], "split"] = "train"
    validation_mask = frame["calendar_week_end"].between(boundaries["validation_start"], boundaries["validation_end"])
    test_mask = frame["calendar_week_end"].between(boundaries["test_start"], boundaries["test_end"])
    frame.loc[validation_mask, "split"] = "validation"
    frame.loc[test_mask, "split"] = "test"
    horizon = int(config["task"]["forecast_horizon_weeks"])
    frame = frame.sort_values(["stock_code", "calendar_week_end"]).reset_index(drop=True)
    grouped = frame.groupby("stock_code", sort=False)
    frame["target_close"] = grouped["model_close"].shift(-horizon)
    frame["target_return"] = frame["target_close"] / frame["model_close"] - 1.0
    frame["target_direction"] = pd.Series(
        np.where(frame["target_return"].notna(), (frame["target_return"] > 0).astype(np.int8), np.nan),
        index=frame.index,
    )
    frame["target_date"] = grouped["calendar_week_end"].shift(-horizon)
    target_split = grouped["split"].shift(-horizon)
    lookback = int(config["task"]["lookback_weeks"])
    frame["history_weeks_available"] = frame.groupby("stock_code", sort=False).cumcount() + 1
    frame["sample_eligible"] = (
        frame["split"].isin(["train", "validation", "test"])
        & frame["split"].eq(target_split)
        & frame["target_close"].notna()
        & frame["history_weeks_available"].ge(lookback)
    )
    return frame, boundaries


def make_calendar(frame: pd.DataFrame, universe_size: int) -> pd.DataFrame:
    calendar = frame.groupby("calendar_week_end", as_index=False).agg(
        observed_stock_count=("stock_code", "nunique"),
        actual_trade_date_min=("trade_date", "min"),
        actual_trade_date_max=("trade_date", "max"),
        friday_observation_count=("is_friday_observation", "sum"),
    )
    calendar["universe_size"] = universe_size
    calendar["cross_section_coverage"] = calendar["observed_stock_count"] / max(universe_size, 1)
    calendar["is_short_holiday_week"] = calendar["actual_trade_date_max"].dt.dayofweek.lt(4)
    return calendar


def write_frame(frame: pd.DataFrame, base_path: Path, output_format: str) -> None:
    if output_format == "parquet":
        frame.to_parquet(base_path.with_suffix(".parquet"), index=False)
    elif output_format == "csv.gz":
        frame.to_csv(base_path.with_suffix(".csv.gz"), index=False, encoding="utf-8-sig", compression="gzip")
    else:
        raise ValueError(f"unsupported output format: {output_format}")


def build(config_path: Path, overwrite: bool = False) -> Path:
    config = read_config(config_path)
    source_root = Path(config["source"]["root"])
    output_root = Path(config["output"]["root"])
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parents[1] / output_root
    allow_overwrite = overwrite or bool(config["output"].get("overwrite"))
    if output_root.exists() and any(output_root.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    files = discover_files(source_root)
    manifest = pd.DataFrame(inspect_file(path) for path in files)
    selected_codes = choose_universe(manifest, config)
    selected_manifest = manifest[manifest["stock_code"].isin(selected_codes)].copy()
    hash_map = selected_manifest.set_index("stock_code")["sha256"].to_dict()
    path_map = {path.stem: path for path in files}
    prices = pd.concat([load_price_file(path_map[code], hash_map[code]) for code in selected_codes], ignore_index=True)
    prices = apply_date_range(prices, config)
    quality = validate_prices(prices)
    fatal_fields = [
        "duplicate_stock_date", "bad_dates", "missing_numeric_cells",
        "invalid_ohlc_rows", "nonpositive_price_rows",
    ]
    failures = [field for field in fatal_fields if quality[field]]
    if failures:
        raise ValueError(f"fatal raw data quality failures: {failures}")

    stock_basic, stock_basic_manifest = load_stock_basic(config)
    prices = merge_stock_basic(prices, stock_basic)
    baostock_weekly, baostock_manifest = load_baostock_weekly(config)
    prices = merge_adjusted_prices(
        prices,
        baostock_weekly,
        required=bool(config.get("baostock", {}).get("required_for_model")),
    )
    prices = add_calendar_and_features(prices)
    csmar_weekly, csmar_manifest = load_csmar_weekly(config)
    prices = merge_csmar_weekly(prices, csmar_weekly)
    prices = merge_text(prices, load_text(config))
    prices, boundaries = assign_splits(prices, config)
    calendar = make_calendar(prices, len(selected_codes))
    coverage = calendar.set_index("calendar_week_end")["cross_section_coverage"]
    prices["cross_section_coverage"] = prices["calendar_week_end"].map(coverage)
    prices["cross_section_eligible"] = prices["cross_section_coverage"].ge(
        float(config["task"]["minimum_cross_section_coverage"])
    )
    prices["sample_eligible"] &= prices["cross_section_eligible"]
    output_format = config["output"]["format"]
    write_frame(manifest, output_root / "source_manifest", output_format)
    if not stock_basic.empty:
        write_frame(stock_basic, output_root / "stock_basic", output_format)
    write_frame(calendar, output_root / "weekly_calendar", output_format)
    write_frame(prices, output_root / "panel", output_format)
    for split_name in ("train", "validation", "test"):
        subset = prices[(prices["split"] == split_name) & prices["sample_eligible"]].copy()
        write_frame(subset, output_root / split_name, output_format)

    metadata = {
        "dataset_version": output_root.name,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": stable_json_hash(config),
        "source": config["source"],
        "stock_basic_source": stock_basic_manifest,
        "baostock_source": baostock_manifest,
        "baostock_coverage_selected_rows": float(prices["baostock_available"].mean()),
        "model_price_basis": "BaoStock前复权（九章量化未复权价格保留用于审计）",
        "csmar_source": csmar_manifest,
        "csmar_weekly_coverage_selected_rows": float(prices["csmar_weekly_available"].mean()),
        "text_event_rows_in_panel": int(prices["text_count"].sum()),
        "stock_basic_coverage_selected_rows": float(prices["stock_basic_available"].mean()),
        "rows_before_listing_date": int(prices.get("before_listing_date", pd.Series(dtype=bool)).sum()),
        "text_enabled": bool(config["text"].get("enabled")),
        "selected_stock_count": len(selected_codes),
        "selected_stocks": selected_codes,
        "raw_file_count": len(files),
        "selected_raw_file_count": len(selected_manifest),
        "quality_before_features": quality,
        "split_boundaries": {key: value.date().isoformat() for key, value in boundaries.items()},
        "row_counts": prices["split"].value_counts().to_dict(),
        "eligible_sample_counts": prices.loc[prices["sample_eligible"], "split"].value_counts().to_dict(),
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "pandas": pd.__version__, "numpy": np.__version__,
        },
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with (output_root / "selected_stocks.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(selected_codes) + "\n")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace files in the configured version directory")
    args = parser.parse_args()
    print(build(args.config, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
