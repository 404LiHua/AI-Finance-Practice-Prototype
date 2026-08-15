from __future__ import annotations

"""Materialize the frozen C0 market-volatility scale feature without FRESH labels.

The calculation is a literal implementation of the market portion of the
archived RG feature builder.  It is accepted only if it exactly reproduces the
controlled 2018--2023 weekly panel before any FRESH row is written.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RG_DATA_REFERENCE_SHA256 = "272753fe9b4331b555a6e2a1ede3178f891ffe9168e86fd4887cf81a7939bdad"
RG_MATERIALIZER_REFERENCE_SHA256 = "73ffa566243dea07f04ab3219069cb6c0f20af9a157e55b86cfef744ebc22880"
FRESH_AVAILABILITY_REFERENCE_SHA256 = "519a9e86c2f46990994c4b13c478e4ccd580bf1cd2f37281afdf4a664e6cac34"
SELECTED_SHA256 = "7522a6053cd143f0046895713b0f66f76a30b15d9ff8ebb8410dc27b0da67f5c"
DELISTED_SHA256 = "fbed94bfc56429e0b7d5a499fbe40f562983e4e0abcc30f67b1124ecb8e318d8"
DEVELOPMENT_WEEKLY_SHA256 = "4633c51055154309a9af766ea51c75f545783c82f4046261cc211f6a8449815f"
DEVELOPMENT_SAMPLES_SHA256 = "60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6"
FRESH_SHA256 = {
    "fresh1": "7706f1dbeebc1e065fbf55266443393a87adbf54702c4162c7d4c02841b30226",
    "fresh2": "f55914c7527df38c95fc294e29d6d9644f91c1e6e6c26ebe6184eaa39b6b61e4",
    "fresh3": "dc4b7f60157b35054a845c2223f6e50a680801a9fbbbf1bd1f371ccb4bedb688",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_daily(path: Path, code: str, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, str]:
    """Use the archived FRESH reader's positional source contract exactly."""
    # Column 15 is the source's close-price-before-adjustment field.  It is
    # the field bound to ``model_close`` by the archived RG1_4 materializer;
    # column 2 is the unadjusted close used by the separate FRESH target path.
    raw = pd.read_csv(path, usecols=[1, 15])
    raw.columns = ["trade_date", "close"]
    raw.trade_date = pd.to_datetime(raw.trade_date, errors="coerce").dt.normalize()
    raw.close = pd.to_numeric(raw.close, errors="coerce")
    raw = raw[raw.trade_date.notna() & (raw.trade_date <= calendar.max())].sort_values("trade_date", kind="mergesort").drop_duplicates("trade_date", keep="last")
    raw_hash = sha256(path)
    weekly = raw.assign(canonical_week=raw.trade_date.dt.to_period("W-FRI").dt.end_time.dt.normalize()).groupby("canonical_week", as_index=True, sort=True).tail(1)
    weekly = weekly.set_index("canonical_week").reindex(calendar)
    result = weekly[["close"]].rename(columns={"close": "model_close"}).reset_index(names="trade_date")
    result["stock_code"] = code
    return result, raw_hash


def build_market_state(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.sort_values(["stock_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    frame["return_1w"] = frame.groupby("stock_code", sort=False)["model_close"].pct_change(fill_method=None)
    market = frame[frame.universe_member_pit.astype(bool)].groupby("trade_date", sort=True)["return_1w"].mean().rename("market_return_1w").reset_index()
    market["market_volatility_4w"] = market.market_return_1w.rolling(4, min_periods=4).std(ddof=0)
    return market


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-root", type=Path, required=True)
    parser.add_argument("--selected-universe", type=Path, required=True)
    parser.add_argument("--delisted-audit", type=Path, required=True)
    parser.add_argument("--development-weekly-panel", type=Path, required=True)
    parser.add_argument("--development-samples", type=Path, required=True)
    parser.add_argument("--fresh1", type=Path, required=True)
    parser.add_argument("--fresh2", type=Path, required=True)
    parser.add_argument("--fresh3", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    daily_root = args.daily_root.resolve(); selected_path = args.selected_universe.resolve(); delisted_path = args.delisted_audit.resolve()
    weekly_path = args.development_weekly_panel.resolve(); samples_path = args.development_samples.resolve(); fresh_paths = {"fresh1": args.fresh1.resolve(), "fresh2": args.fresh2.resolve(), "fresh3": args.fresh3.resolve()}; output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    if any(token in str(path).lower() for path in (*fresh_paths.values(), output) for token in ("screening", "final", "sealed_holdout")):
        raise RuntimeError("prohibited holdout path")
    for path, expected, label in ((selected_path, SELECTED_SHA256, "selected universe"), (delisted_path, DELISTED_SHA256, "delisted audit"), (weekly_path, DEVELOPMENT_WEEKLY_SHA256, "development weekly panel"), (samples_path, DEVELOPMENT_SAMPLES_SHA256, "development samples")):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"{label} identity mismatch")
    for name, path in fresh_paths.items():
        if not path.is_file() or sha256(path) != FRESH_SHA256[name]:
            raise RuntimeError(f"FRESH identity mismatch: {name}")
    selected = pd.read_csv(selected_path, usecols=["stock_code", "selection_rank"], dtype={"stock_code": str}).sort_values("selection_rank", kind="mergesort")
    delisted = pd.read_csv(delisted_path, usecols=["stock_code"], dtype={"stock_code": str})
    if len(selected) != 300 or len(delisted) != 12 or selected.stock_code.duplicated().any() or delisted.stock_code.duplicated().any() or set(selected.stock_code) & set(delisted.stock_code):
        raise RuntimeError("frozen market-universe contract failure")
    # This purposefully opens only identity dates, never FRESH targets/labels.
    fresh_dates = []
    for path in fresh_paths.values():
        frame = pd.read_csv(path, usecols=["trade_date", "sample_key_sha256"], dtype={"sample_key_sha256": str})
        frame.trade_date = pd.to_datetime(frame.trade_date, errors="raise").dt.normalize()
        if frame.sample_key_sha256.duplicated().any():
            raise RuntimeError(f"duplicate FRESH keys: {path.name}")
        fresh_dates.extend(frame.trade_date.unique())
    requested_dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(fresh_dates).unique()))
    frozen_membership = pd.read_csv(weekly_path, usecols=["trade_date", "stock_code", "universe_member_pit"], dtype={"stock_code": str})
    frozen_membership.trade_date = pd.to_datetime(frozen_membership.trade_date, errors="raise").dt.normalize()
    if frozen_membership.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("controlled development membership keys are not unique")
    calendar = pd.date_range("2018-06-08", requested_dates.max(), freq="W-FRI")
    panels, raw_hashes = [], {}
    selected_codes = set(selected.stock_code)
    for code in [*selected.stock_code, *delisted.stock_code]:
        path = daily_root / f"{code}.csv"
        if not path.is_file():
            raise RuntimeError(f"missing frozen market-universe daily source: {code}")
        panel, raw_hash = read_daily(path, code, calendar)
        panel = panel.merge(frozen_membership, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
        # After the development panel, the frozen selected 300 remain the
        # market universe.  All 12 additions were delisted by 2023-04-03 and
        # therefore have no post-development market return to contribute.
        panel["universe_member_pit"] = panel.universe_member_pit.fillna(code in selected_codes).astype(bool)
        panels.append(panel); raw_hashes[code] = raw_hash
    market = build_market_state(pd.concat(panels, ignore_index=True))
    development = pd.read_csv(weekly_path, usecols=["trade_date", "market_volatility_4w"])
    development.trade_date = pd.to_datetime(development.trade_date, errors="raise").dt.normalize()
    development = development.drop_duplicates("trade_date", keep="first")
    replay = development.merge(market[["trade_date", "market_volatility_4w"]], on="trade_date", how="left", validate="one_to_one", suffixes=("_frozen", "_replayed"))
    # The archived panel has two unavailable observations on its first origin
    # date (2018-06-08), whereas the retained raw-daily source now contains
    # them.  This changes only the first calculable four-week state
    # (2018-07-06).  Require exact replay after that unavoidable warm-up
    # boundary; FRESH starts more than 250 weeks later.
    warmup_end = pd.Timestamp("2018-07-06")
    replay_after_warmup = replay[replay.trade_date > warmup_end].copy()
    equality = np.isclose(replay_after_warmup.market_volatility_4w_frozen.to_numpy(float), replay_after_warmup.market_volatility_4w_replayed.to_numpy(float), equal_nan=True, rtol=0.0, atol=1e-14)
    if not bool(np.all(equality)):
        finite = replay[np.isfinite(replay.market_volatility_4w_frozen) & np.isfinite(replay.market_volatility_4w_replayed)]
        maximum_error = float(np.max(np.abs(finite.market_volatility_4w_frozen - finite.market_volatility_4w_replayed))) if len(finite) else None
        raise RuntimeError(f"archived raw-daily replay fails controlled development market volatility; max_error={maximum_error}")
    result = market[market.trade_date.isin(requested_dates)].copy()
    if len(result) != len(requested_dates) or result.trade_date.duplicated().any():
        raise RuntimeError("FRESH market-state coverage failure")
    result["source_trade_date"] = result.trade_date
    output.mkdir(parents=True)
    state_path = output / "fresh_market_volatility_4w_v1.csv.gz"
    result[["trade_date", "market_volatility_4w", "source_trade_date"]].to_csv(state_path, index=False, compression={"method": "gzip", "mtime": 0}, lineterminator="\n")
    input_hashes = {"selected_universe": sha256(selected_path), "delisted_audit": sha256(delisted_path), "development_weekly_panel": sha256(weekly_path), "development_samples": sha256(samples_path), **{name: sha256(path) for name, path in fresh_paths.items()}}
    receipt = {
        "node_id": "WP10_FRESH_MARKET_VOLATILITY_SUPPLEMENT_FREEZE_V1",
        "status": "FROZEN_BEFORE_WP10_FRESH_PREDICTION_SEAL",
        "market_state_sha256": sha256(state_path),
        "development_weekly_panel_sha256": DEVELOPMENT_WEEKLY_SHA256,
        "input_hashes": input_hashes,
        "raw_daily_source_sha256": raw_hashes,
        "reference_source_sha256": {
            "rg_data": RG_DATA_REFERENCE_SHA256,
            "rg1_4_materializer": RG_MATERIALIZER_REFERENCE_SHA256,
            "fresh_availability_reader": FRESH_AVAILABILITY_REFERENCE_SHA256,
        },
        "market_universe": {
            "selected_stocks": int(len(selected)), "delisted_stocks": int(len(delisted)),
            "membership_rule": "controlled weekly-panel membership is replayed on development dates; thereafter the archived selected 300 are members. All 12 additions were delisted by 2023-04-03 and have no post-development market return",
        },
        "development_replay": {
            "dates_total": int(len(replay)), "warmup_end_exclusive": warmup_end.date().isoformat(),
            "dates_exact_after_warmup": int(len(replay_after_warmup)), "exact_match_after_warmup_within_machine_precision": True,
            "maximum_absolute_error_all_dates": float(np.nanmax(np.abs(replay.market_volatility_4w_frozen - replay.market_volatility_4w_replayed))),
            "maximum_absolute_error_after_warmup": float(np.nanmax(np.abs(replay_after_warmup.market_volatility_4w_frozen - replay_after_warmup.market_volatility_4w_replayed))),
            "warmup_exception": "2018-07-06 only: two 2018-06-08 source-panel observations were unavailable in the archived panel but are present in retained daily raw files; no post-warmup value differs",
        },
        "fresh_dates": {"count": int(len(requested_dates)), "start": requested_dates.min().date().isoformat(), "end": requested_dates.max().date().isoformat()},
        "fresh_labels_read": False, "fresh_metrics_read": False, "screening_read": False, "final_read": False, "gpu_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "WP10_FRESH_MARKET_VOLATILITY_SUPPLEMENT_FREEZE.json", receipt)
    print(json.dumps({"status": receipt["status"], "output_root": str(output), "fresh_dates": receipt["fresh_dates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


