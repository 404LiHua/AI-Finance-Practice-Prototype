from __future__ import annotations

"""Materialize a keyed, label-free C1 future scale panel.

The C1 model needs ``market_volatility_4w`` beside every
``(trade_date, stock_code)`` key.  The underlying state is market-level, so
the value is broadcast only after it has been computed from the frozen
300-stock universe.  This script records that rule explicitly and proves the
calculation by replaying the archived development weekly panel before it
writes the future panel.

This script never opens target labels, FRESH payloads, SCREENING, FINAL, or
the production registry.  It is CPU/I/O work; a small worker pool is used so
that the GPU remains free for a single controlled research training job.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BAN = ("fresh", "screening", "final", "sealed_holdout")
EXPECTED_SELECTED_SHA256 = "7522a6053cd143f0046895713b0f66f76a30b15d9ff8ebb8410dc27b0da67f5c"
EXPECTED_DELISTED_SHA256 = "fbed94bfc56429e0b7d5a499fbe40f562983e4e0abcc30f67b1124ecb8e318d8"
EXPECTED_DEVELOPMENT_WEEKLY_SHA256 = "4633c51055154309a9af766ea51c75f545783c82f4046261cc211f6a8449815f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_daily(path: Path, code: str, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, str]:
    # Column 15 is the archived RG model_close binding: close price before
    # adjustment.  Column 1 is the source trade date.
    raw = pd.read_csv(path, usecols=[1, 15])
    raw.columns = ["trade_date", "close"]
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw[raw["trade_date"].notna() & raw["close"].notna()]
    raw = raw[raw["trade_date"] <= calendar.max()]
    raw = raw.sort_values("trade_date", kind="mergesort").drop_duplicates("trade_date", keep="last")
    weekly = (
        raw.assign(canonical_week=raw["trade_date"].dt.to_period("W-FRI").dt.end_time.dt.normalize())
        .groupby("canonical_week", as_index=True, sort=True)
        .tail(1)
        .set_index("canonical_week")
        .reindex(calendar)
    )
    result = weekly[["close"]].rename(columns={"close": "model_close"}).reset_index(names="trade_date")
    result["stock_code"] = code
    return result, sha256(path)


def build_market_state(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.sort_values(["stock_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    frame["return_1w"] = frame.groupby("stock_code", sort=False)["model_close"].pct_change(fill_method=None)
    market = (
        frame[frame["universe_member_pit"].astype(bool)]
        .groupby("trade_date", sort=True)["return_1w"]
        .mean()
        .rename("market_return_1w")
        .reset_index()
    )
    market["market_volatility_4w"] = market["market_return_1w"].rolling(4, min_periods=4).std(ddof=0)
    return market


def load_shadow_keys(path: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, dtype={"stock_code": str})
    required = {"trade_date", "stock_code"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"shadow input missing key columns: {missing}")
    frame = frame[["trade_date", "stock_code"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["stock_code"] = frame["stock_code"].astype(str)
    dates = frame["trade_date"].drop_duplicates().sort_values().tolist()
    if len(dates) != 1:
        raise RuntimeError(f"future shadow input must contain exactly one origin date, got {len(dates)}")
    if frame.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("future shadow input has duplicated stock keys")
    if len(frame) != 300 or frame["stock_code"].nunique() != 300:
        raise RuntimeError(f"future shadow input must contain exactly 300 unique keys, got {len(frame)}")
    return frame, pd.Timestamp(dates[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-input", required=True, type=Path)
    parser.add_argument("--daily-root", required=True, type=Path)
    parser.add_argument("--selected-universe", required=True, type=Path)
    parser.add_argument("--delisted-audit", required=True, type=Path)
    parser.add_argument("--development-weekly-panel", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    shadow_path = args.shadow_input.resolve()
    daily_root = args.daily_root.resolve()
    selected_path = args.selected_universe.resolve()
    delisted_path = args.delisted_audit.resolve()
    development_path = args.development_weekly_panel.resolve()
    output_root = args.output_root.resolve()
    paths = (shadow_path, daily_root, selected_path, delisted_path, development_path, output_root)
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite output: {output_root}")
    if any(token in str(path).lower() for path in paths for token in BAN):
        raise RuntimeError("prohibited holdout/fresh path token")
    if not shadow_path.is_file() or not daily_root.is_dir() or not selected_path.is_file() or not delisted_path.is_file() or not development_path.is_file():
        raise RuntimeError("one or more required inputs are missing")
    if sha256(selected_path) != EXPECTED_SELECTED_SHA256:
        raise RuntimeError("selected-universe SHA256 mismatch")
    if sha256(delisted_path) != EXPECTED_DELISTED_SHA256:
        raise RuntimeError("delisted-audit SHA256 mismatch")
    if sha256(development_path) != EXPECTED_DEVELOPMENT_WEEKLY_SHA256:
        raise RuntimeError("development weekly-panel SHA256 mismatch")
    if not (1 <= args.workers <= 8):
        raise RuntimeError("workers must be between 1 and 8")

    shadow, target_date = load_shadow_keys(shadow_path)
    selected = pd.read_csv(selected_path, usecols=["selection_rank", "stock_code"], dtype={"stock_code": str})
    selected = selected.sort_values("selection_rank", kind="mergesort").reset_index(drop=True)
    selected_codes = selected["stock_code"].astype(str).tolist()
    if set(selected_codes) != set(shadow["stock_code"]):
        raise RuntimeError("shadow stock keys do not exactly match the frozen selected universe")
    if target_date.dayofweek != 4:
        raise RuntimeError("origin date must be canonical Friday")

    delisted = pd.read_csv(delisted_path, usecols=["stock_code"], dtype={"stock_code": str})
    delisted_codes = delisted["stock_code"].astype(str).tolist()
    if len(delisted_codes) != 12 or len(set(delisted_codes)) != 12 or set(delisted_codes) & set(selected_codes):
        raise RuntimeError("frozen delisted-universe contract failure")

    development = pd.read_csv(
        development_path,
        usecols=["trade_date", "stock_code", "universe_member_pit", "market_volatility_4w"],
        dtype={"stock_code": str},
    )
    development["trade_date"] = pd.to_datetime(development["trade_date"], errors="raise").dt.normalize()
    development["stock_code"] = development["stock_code"].astype(str)
    if development.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("development membership keys are not unique")
    membership = development[["trade_date", "stock_code", "universe_member_pit"]]
    calendar = pd.date_range("2018-06-08", target_date, freq="W-FRI")

    # The 12 historical delisted members are required for exact development
    # replay. They are excluded from the future market state after their
    # frozen membership ends, so they do not change the future broadcast.
    replay_codes = [*selected_codes, *delisted_codes]
    missing = [code for code in replay_codes if not (daily_root / f"{code}.csv").is_file()]
    if missing:
        raise RuntimeError(f"missing daily source files: {missing[:10]} (total={len(missing)})")

    # Limit BLAS/OpenMP fan-out inside each pandas worker; this is deliberately
    # a modest CPU/I/O phase and keeps the GPU untouched.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    panels: list[pd.DataFrame] = []
    raw_hashes: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(read_daily, daily_root / f"{code}.csv", code, calendar): code for code in replay_codes}
        for future in as_completed(futures):
            code = futures[future]
            panel, raw_hash = future.result()
            panels.append(panel)
            raw_hashes[code] = raw_hash
    raw_panel = pd.concat(panels, ignore_index=True)
    raw_panel = raw_panel.merge(membership, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    selected_set = set(selected_codes)
    raw_panel["universe_member_pit"] = raw_panel["universe_member_pit"].fillna(raw_panel["stock_code"].isin(selected_set)).astype(bool)
    market = build_market_state(raw_panel)

    # Development replay: after the known first-origin warm-up exception, the
    # archived formula must agree to machine precision before future output is
    # accepted.
    frozen_dev = development[["trade_date", "market_volatility_4w"]].drop_duplicates("trade_date", keep="first")
    replay = frozen_dev.merge(market[["trade_date", "market_volatility_4w"]], on="trade_date", how="left", suffixes=("_frozen", "_replayed"), validate="one_to_one")
    warmup_end = pd.Timestamp("2018-07-06")
    after = replay[replay["trade_date"] > warmup_end].copy()
    if after.empty:
        raise RuntimeError("development replay produced no post-warm-up rows")
    equal = np.isclose(after["market_volatility_4w_frozen"].to_numpy(float), after["market_volatility_4w_replayed"].to_numpy(float), rtol=0.0, atol=1e-14, equal_nan=True)
    if not bool(np.all(equal)):
        finite = replay[np.isfinite(replay["market_volatility_4w_frozen"]) & np.isfinite(replay["market_volatility_4w_replayed"])]
        max_error = float(np.max(np.abs(finite["market_volatility_4w_frozen"] - finite["market_volatility_4w_replayed"]))) if len(finite) else None
        raise RuntimeError(f"development market-volatility replay failed: max_error={max_error}")

    target_state = market.loc[market["trade_date"].eq(target_date), "market_volatility_4w"]
    if len(target_state) != 1 or not np.isfinite(float(target_state.iloc[0])):
        raise RuntimeError(f"target market state unavailable at {target_date.date()}")
    value = float(target_state.iloc[0])
    result = shadow[["trade_date", "stock_code"]].copy()
    result["market_volatility_4w"] = value
    result["source_trade_date"] = result["trade_date"]
    result = result[["trade_date", "stock_code", "market_volatility_4w", "source_trade_date"]]

    output_root.mkdir(parents=True)
    panel_path = output_root / "WP22_C1_FUTURE_MARKET_SCALE_PANEL.parquet"
    receipt_path = output_root / "WP22_C1_FUTURE_MARKET_SCALE_PANEL_RECEIPT.json"
    result.to_parquet(panel_path, index=False, engine="pyarrow", compression="zstd")
    receipt = {
        "node_id": "WP22_C1_FUTURE_MARKET_SCALE_PANEL_V1",
        "status": "PASS_LABEL_FREE_KEYED_SCALE_PANEL_MATERIALIZED",
        "target_id": "T2_MARKET_RELATIVE_FIXED",
        "origin_date": target_date.date().isoformat(),
        "rows": int(len(result)),
        "unique_stock_codes": int(result["stock_code"].nunique()),
        "market_state_value": value,
        "market_state_formula": "mean weekly pct_change of model_close over frozen 300-stock universe, then rolling 4-week population standard deviation (ddof=0)",
        "key_broadcast_rule": "the single market-level state at origin_date is copied to every one of the 300 frozen stock keys; no per-stock imputation or future fill",
        "input_sha256": {
            "shadow_input": sha256(shadow_path),
            "selected_universe": sha256(selected_path),
            "delisted_audit": sha256(delisted_path),
            "development_weekly_panel": sha256(development_path),
            "daily_sources": raw_hashes,
        },
        "output_sha256": sha256(panel_path),
        "development_replay": {
            "dates_total": int(len(replay)),
            "post_warmup_dates": int(len(after)),
            "exact_after_warmup": True,
            "warmup_exception": "2018-07-06 only, inherited from the archived RG replay contract",
        },
        "fresh_labels_read": False,
        "screening_read": False,
        "final_read": False,
        "target_labels_read": False,
        "production_registry_modified": False,
        "gpu_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "rows": receipt["rows"], "origin_date": receipt["origin_date"], "market_volatility_4w": value, "output": str(panel_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
