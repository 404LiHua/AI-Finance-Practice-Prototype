from __future__ import annotations

"""Create the frozen C0 market-scale feature from daily data without labels."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_weekly_close(path: Path, code: str, origin: pd.Timestamp) -> tuple[pd.DataFrame, str, pd.Timestamp | None]:
    # Column 15 is the archived RG materializer's model-close input.  It is
    # deliberately not the target-return price column, which this tool never
    # opens or derives.
    raw = pd.read_csv(path, usecols=[1, 15])
    raw.columns = ["trade_date", "model_close"]
    raw.trade_date = pd.to_datetime(raw.trade_date, errors="coerce").dt.normalize()
    raw.model_close = pd.to_numeric(raw.model_close, errors="coerce")
    raw = raw[raw.trade_date.notna() & (raw.trade_date <= origin)].sort_values("trade_date", kind="mergesort").drop_duplicates("trade_date", keep="last")
    source_date = raw.trade_date.max() if len(raw) else None
    raw["canonical_week"] = raw.trade_date.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    weekly = raw.groupby("canonical_week", as_index=False, sort=True).tail(1)[["canonical_week", "model_close"]]
    weekly["stock_code"] = code
    return weekly, sha256(path), source_date


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-root", required=True, type=Path)
    parser.add_argument("--selected-universe", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    daily_root, selected_path, protocol_path, output = (args.daily_root.resolve(), args.selected_universe.resolve(), args.protocol.resolve(), args.output_root.resolve())
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_C0_ARTIFACT_MATERIALIZATION_OR_WP11_SHADOW_PREDICTION":
        raise RuntimeError("WP11 protocol is not frozen")
    if sha256(selected_path) != protocol["input_identities"]["selected_universe_sha256"]:
        raise RuntimeError("selected-universe identity mismatch")
    origin = pd.Timestamp(protocol["shadow"]["origin_date"]).normalize()
    cutoff = pd.Timestamp(protocol["shadow"]["incumbent_fit_cutoff_exclusive"]).normalize()
    if origin.weekday() != 4 or origin <= cutoff:
        raise RuntimeError("WP11 origin contract failure")
    selected = pd.read_csv(selected_path, usecols=["stock_code", "selection_rank"], dtype={"stock_code": str}).sort_values("selection_rank", kind="mergesort")
    if len(selected) != int(protocol["shadow"]["frozen_universe_size"]) or selected.stock_code.duplicated().any():
        raise RuntimeError("selected-universe cardinality/key contract failure")
    weekly_parts: list[pd.DataFrame] = []
    raw_hashes: dict[str, str] = {}
    source_dates: dict[str, str | None] = {}
    for code in selected.stock_code:
        path = daily_root / f"{code}.csv"
        if not path.is_file():
            raise RuntimeError(f"missing daily source: {code}")
        weekly, file_hash, source_date = read_weekly_close(path, code, origin)
        weekly_parts.append(weekly); raw_hashes[code] = file_hash
        source_dates[code] = source_date.date().isoformat() if source_date is not None else None
    panel = pd.concat(weekly_parts, ignore_index=True).rename(columns={"canonical_week": "trade_date"})
    calendar = pd.date_range(end=origin, periods=5, freq="W-FRI")
    panel = panel[panel.trade_date.isin(calendar)].copy()
    panel = panel.sort_values(["stock_code", "trade_date"], kind="mergesort")
    panel["return_1w"] = panel.groupby("stock_code", sort=False).model_close.pct_change(fill_method=None)
    market = panel.groupby("trade_date", sort=True).return_1w.mean().rename("market_return_1w").reset_index()
    market["market_volatility_4w"] = market.market_return_1w.rolling(4, min_periods=4).std(ddof=0)
    row = market[market.trade_date.eq(origin)]
    if len(row) != 1 or not np.isfinite(row.market_volatility_4w.to_numpy(float)).all():
        raise RuntimeError("market-volatility warm-up or numeric contract failure")
    current = panel[panel.trade_date.eq(origin)]
    state = row[["trade_date", "market_volatility_4w"]].copy()
    state["source_trade_date"] = origin
    state["market_constituent_observations"] = int(current.model_close.notna().sum())
    state["market_constituent_same_day_observations"] = int(sum(date == origin.date().isoformat() for date in source_dates.values()))
    output.mkdir(parents=True)
    state_path = output / "WP11_LABEL_FREE_MARKET_VOLATILITY_4W.parquet"
    state.to_parquet(state_path, index=False, engine="pyarrow", compression="zstd")
    receipt = {
        "node_id": "AA_GFMNET_WP11_LABEL_FREE_MARKET_VOLATILITY_V1",
        "status": "PASS_LABEL_FREE_WP11_MARKET_STATE_SEALED",
        "protocol_sha256": sha256(protocol_path),
        "origin_date": origin.date().isoformat(),
        "market_state_sha256": sha256(state_path),
        "selected_universe_sha256": sha256(selected_path),
        "daily_source_sha256": raw_hashes,
        "source_trade_date_by_stock": source_dates,
        "market_constituent_observations": int(state.market_constituent_observations.iloc[0]),
        "market_constituent_same_day_observations": int(state.market_constituent_same_day_observations.iloc[0]),
        "target_labels_read": False,
        "fresh_labels_read": False,
        "metrics_read": False,
        "model_trained": False,
        "gpu_used": False,
        "automatic_trading": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "WP11_MARKET_STATE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "market_volatility_4w": float(state.market_volatility_4w.iloc[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()


