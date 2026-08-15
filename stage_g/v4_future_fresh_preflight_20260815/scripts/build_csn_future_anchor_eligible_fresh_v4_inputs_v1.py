from __future__ import annotations

"""Fail-closed V4 materializer for the pre-registered Monday-09:30 H4 window.

This is deliberately independent of the legacy V1/V3 FRESH builders.  It can only
run after the V4 delivery has been frozen and attested; it writes labels straight
to the sealed parquet output and never prints their values, row count, returns, or
any performance statistic.  Do not use this tool for a dry run before 2026-09-11.
"""

import argparse
import hashlib
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ORIGINS = tuple(pd.to_datetime((
    "2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10",
    "2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07",
)))
ORIGIN_TEXT = tuple(day.date().isoformat() for day in ORIGINS)
LAST_SETTLEMENT_DATE = date(2026, 9, 11)
DELIVERY_STATUS = "PASS_V4_DATA_DELIVERY_FROZEN_FOR_SEALED_MATERIALIZATION"
LABEL_PROTOCOL = "TEXTCU_V2_H4_MONDAY_0930_V1"
NUMERIC_FIELDS = ("return_1w", "log_return_1w", "return_vol_4", "return_vol_12", "model_close", "model_volume_hands")
TECH_FIELDS = (
    "momentum_20d", "momentum_60d", "momentum_120d", "realized_volatility_20d",
    "realized_volatility_60d", "downside_volatility_60d", "current_drawdown_60d",
    "rsi_14", "macd_scaled", "bollinger_position_20", "amihud_20d",
    "volume_ratio_20d_60d", "intraday_range_mean_20d", "technical_available",
)
FUND_FIELDS = (
    "log_total_assets", "debt_to_assets", "equity_to_assets", "return_on_assets",
    "net_margin", "asset_turnover", "revenue_yoy", "profit_yoy", "asset_growth_yoy",
    "leverage_change_yoy", "report_age_anchor_days", "has_fundamental_event",
)
REQUIRED_DAILY = (
    "交易日期", "成交量(手)", "成交额(千元)", "收盘价前复权", "最高价前复权",
    "最低价前复权", "RSI_12", "MACD_DIF(基于前复权价格计算)", "MACD_DEA",
    "BOLL_UPPER", "BOLL_MID", "BOLL_LOWER",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def normalized_codes(frame: pd.DataFrame) -> list[str]:
    if "stock_code" not in frame.columns:
        raise RuntimeError("FAIL_CLOSED_V4_UNIVERSE_SCHEMA")
    result = frame["stock_code"].astype(str).str.strip().str.upper().tolist()
    if len(result) != len(set(result)) or not 200 <= len(result) <= 300:
        raise RuntimeError("FAIL_CLOSED_V4_UNIVERSE_COUNT_OR_DUPLICATE")
    if any(not pd.Series([code]).str.fullmatch(r"[0-9]{6}\.(SZ|SH|BJ)").iloc[0] for code in result):
        raise RuntimeError("FAIL_CLOSED_V4_UNIVERSE_STOCK_CODE")
    return result


def load_universe(path: Path) -> list[str]:
    frame = read_table(path)
    if "origin_date" not in frame.columns:
        return normalized_codes(frame)
    frame = frame.loc[:, ["origin_date", "stock_code"]].copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="coerce").dt.date.astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str).str.strip().str.upper()
    expected = set(ORIGIN_TEXT)
    if frame["origin_date"].isna().any() or set(frame["origin_date"]) != expected or frame.duplicated().any():
        raise RuntimeError("FAIL_CLOSED_V4_UNIVERSE_ORIGIN_GRID")
    per_origin = [tuple(sorted(frame.loc[frame.origin_date.eq(origin), "stock_code"])) for origin in ORIGIN_TEXT]
    if any(items != per_origin[0] for items in per_origin[1:]):
        raise RuntimeError("FAIL_CLOSED_V4_UNIVERSE_NOT_CONSTANT")
    return normalized_codes(pd.DataFrame({"stock_code": per_origin[0]}))


def load_delivery_attestation(path: Path, daily_manifest: Path, fundamentals: Path, universe: Path, materialization_date: date) -> dict:
    if materialization_date < LAST_SETTLEMENT_DATE:
        raise RuntimeError("FAIL_CLOSED_V4_NOT_YET_MATERIALIZABLE")
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {"status", "delivery_id", "origin_dates", "daily_manifest_sha256", "fundamental_events_sha256", "universe_sha256", "labels_custody_state"}
    if not required.issubset(record) or record["status"] != DELIVERY_STATUS:
        raise RuntimeError("FAIL_CLOSED_V4_DELIVERY_ATTESTATION")
    if record["origin_dates"] != list(ORIGIN_TEXT):
        raise RuntimeError("FAIL_CLOSED_V4_ATTESTATION_ORIGINS")
    if record["labels_custody_state"] != "SEALED_NOT_READ_BY_MATERIALIZER":
        raise RuntimeError("FAIL_CLOSED_V4_LABEL_CUSTODY")
    checks = {
        "daily_manifest_sha256": sha256(daily_manifest),
        "fundamental_events_sha256": sha256(fundamentals),
        "universe_sha256": sha256(universe),
    }
    if any(record[key] != value for key, value in checks.items()):
        raise RuntimeError("FAIL_CLOSED_V4_DELIVERY_HASH_MISMATCH")
    return record


def load_daily_manifest(path: Path, codes: list[str]) -> dict[str, str]:
    manifest = pd.read_csv(path, encoding="utf-8-sig")
    if not {"stock_code", "sha256"}.issubset(manifest.columns):
        raise RuntimeError("FAIL_CLOSED_V4_DAILY_MANIFEST_SCHEMA")
    manifest["stock_code"] = manifest["stock_code"].astype(str).str.strip().str.upper()
    if manifest.duplicated("stock_code").any() or set(manifest.stock_code) != set(codes):
        raise RuntimeError("FAIL_CLOSED_V4_DAILY_MANIFEST_KEY_DOMAIN")
    hashes = manifest.set_index("stock_code")["sha256"].astype(str).to_dict()
    if any(len(value) != 64 for value in hashes.values()):
        raise RuntimeError("FAIL_CLOSED_V4_DAILY_MANIFEST_HASH")
    return hashes


def load_daily(path: Path, expected_sha: str) -> pd.DataFrame:
    if not path.is_file() or sha256(path) != expected_sha:
        raise RuntimeError(f"FAIL_CLOSED_V4_DAILY_SOURCE_HASH_{path.stem}")
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing = set(REQUIRED_DAILY).difference(header)
    if missing:
        raise RuntimeError(f"FAIL_CLOSED_V4_DAILY_SCHEMA_{path.stem}")
    daily = pd.read_csv(path, usecols=list(REQUIRED_DAILY), encoding="utf-8-sig")
    daily = daily.rename(columns={"交易日期": "trade_date", "成交量(手)": "volume_hands", "成交额(千元)": "amount_k", "收盘价前复权": "close_qfq", "最高价前复权": "high_qfq", "最低价前复权": "low_qfq", "RSI_12": "rsi_12", "MACD_DIF(基于前复权价格计算)": "macd_dif", "MACD_DEA": "macd_dea", "BOLL_UPPER": "boll_upper", "BOLL_MID": "boll_mid", "BOLL_LOWER": "boll_lower"})
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.normalize()
    for column in daily.columns.drop("trade_date"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily = daily.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
    if daily.empty or (daily.close_qfq.dropna() <= 0).any():
        raise RuntimeError(f"FAIL_CLOSED_V4_DAILY_VALUE_{path.stem}")
    return daily


def assert_coverage(daily: pd.DataFrame, code: str) -> None:
    for origin in ORIGINS:
        history = daily.loc[daily.trade_date < origin]
        target = daily.loc[daily.trade_date >= origin]
        # H4 is four trading days *after* the Monday decision time: the normal
        # target is Friday, so origin day plus four subsequent trading days are
        # required in the frozen daily snapshot.
        if len(history) < 120 or len(target) < 5:
            raise RuntimeError(f"FAIL_CLOSED_V4_DAILY_COVERAGE_{code}_{origin.date()}")
        if not np.isfinite(history.tail(120).close_qfq).all() or not np.isfinite(target.head(4).close_qfq).all():
            raise RuntimeError(f"FAIL_CLOSED_V4_PRICE_COVERAGE_{code}_{origin.date()}")


def numeric_features(daily: pd.DataFrame) -> dict[str, np.ndarray]:
    result = np.full((len(ORIGINS), 8, len(NUMERIC_FIELDS)), np.nan, dtype=np.float32)
    weekly = daily.loc[:, ["trade_date", "close_qfq", "volume_hands"]].copy()
    weekly["week_end"] = weekly.trade_date.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    weekly = weekly.drop_duplicates("week_end", keep="last").set_index("week_end").sort_index()
    weekly["return_1w"] = weekly.close_qfq.pct_change()
    weekly["log_return_1w"] = np.log(weekly.close_qfq).diff()
    weekly["return_vol_4"] = weekly.return_1w.rolling(4, min_periods=4).std()
    weekly["return_vol_12"] = weekly.return_1w.rolling(12, min_periods=12).std()
    weekly["model_close"] = weekly.close_qfq
    weekly["model_volume_hands"] = weekly.volume_hands
    for index, origin in enumerate(ORIGINS):
        source = weekly.loc[weekly.index < origin, list(NUMERIC_FIELDS)].tail(8)
        if len(source) != 8:
            raise RuntimeError("FAIL_CLOSED_V4_NUMERIC_SEQUENCE")
        result[index] = source.to_numpy(dtype=np.float32)
    return {"numeric": result}


def technical_row(daily: pd.DataFrame, origin: pd.Timestamp) -> dict[str, float | bool]:
    frame = daily.loc[daily.trade_date < origin].tail(120).copy()
    close, high, low, volume, amount = (frame[column].to_numpy(dtype=float) for column in ("close_qfq", "high_qfq", "low_qfq", "volume_hands", "amount_k"))
    if len(frame) < 120 or not np.isfinite(close).all() or (close <= 0).any():
        raise RuntimeError("FAIL_CLOSED_V4_TECHNICAL_COVERAGE")
    def momentum(days: int) -> float: return float(close[-1] / close[-days] - 1.0)
    def volatility(days: int, downside: bool = False) -> float:
        values = np.diff(np.log(close[-(days + 1):])); values = values[values < 0] if downside else values
        return float(np.std(values)) if len(values) else 0.0
    trailing60 = close[-60:]; peaks = np.maximum.accumulate(trailing60)
    delta = np.diff(close[-15:]); gains, losses = delta.clip(min=0).mean(), (-delta.clip(max=0)).mean()
    rsi = 100.0 if losses == 0 else 100.0 - 100.0 / (1.0 + gains / losses)
    upper, lower = frame.boll_upper.iloc[-1], frame.boll_lower.iloc[-1]
    boll = 0.5 if not np.isfinite(upper - lower) or upper == lower else float((close[-1] - lower) / (upper - lower))
    returns20 = np.diff(np.log(close[-21:]))
    range20 = (high[-20:] - low[-20:]) / ((high[-20:] + low[-20:] + close[-20:]) / 3.0 + 1e-12)
    return {
        "momentum_20d": momentum(20), "momentum_60d": momentum(60), "momentum_120d": momentum(120),
        "realized_volatility_20d": volatility(20), "realized_volatility_60d": volatility(60), "downside_volatility_60d": volatility(60, True),
        "current_drawdown_60d": float(np.min((trailing60 - peaks) / peaks)), "rsi_14": float(rsi),
        "macd_scaled": float((frame.macd_dif.iloc[-1] - frame.macd_dea.iloc[-1]) / close[-1]), "bollinger_position_20": boll,
        "amihud_20d": float(np.mean(np.abs(returns20) / (amount[-20:] + 1e-12))),
        "volume_ratio_20d_60d": float(np.mean(volume[-20:]) / (np.mean(volume[-60:]) + 1e-12)),
        "intraday_range_mean_20d": float(np.mean(range20)), "technical_available": True,
    }


def fundamental_panel(events: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    required = {"stock_code", "available_at"}.union(FUND_FIELDS[:-1])
    if not required.issubset(events.columns):
        raise RuntimeError("FAIL_CLOSED_V4_FUNDAMENTAL_SCHEMA")
    events = events.copy(); events.stock_code = events.stock_code.astype(str).str.strip().str.upper()
    events["available_at"] = pd.to_datetime(events.available_at, errors="coerce")
    if events.available_at.isna().any():
        raise RuntimeError("FAIL_CLOSED_V4_FUNDAMENTAL_AVAILABLE_AT")
    rows: list[dict[str, object]] = []
    for origin in ORIGINS:
        cutoff = origin + pd.Timedelta(hours=9, minutes=30)
        available = events.loc[events.available_at <= cutoff].sort_values("available_at")
        latest = available.loc[available.stock_code.isin(codes)].groupby("stock_code", as_index=False).tail(1).set_index("stock_code")
        for code in codes:
            row: dict[str, object] = {"origin_date": origin.date().isoformat(), "stock_code": code}
            if code in latest.index:
                for field in FUND_FIELDS[:-1]: row[field] = latest.at[code, field]
                row["has_fundamental_event"] = True
            else:
                for field in FUND_FIELDS[:-1]: row[field] = np.nan
                row["has_fundamental_event"] = False
            rows.append(row)
    return pd.DataFrame(rows, columns=["origin_date", "stock_code", *FUND_FIELDS])


def sealed_labels(daily_by_code: dict[str, pd.DataFrame], source_hashes: dict[str, str], codes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for origin in ORIGINS:
        for code in codes:
            daily = daily_by_code[code]
            before = daily.loc[daily.trade_date < origin]
            after = daily.loc[daily.trade_date >= origin]
            row: dict[str, object] = {"origin_date": origin.date().isoformat(), "stock_code": code, "h4_return": np.nan, "target_horizon_trading_days": 4, "anchor_trade_date": None, "first_trade_on_or_after_origin": None, "target_trade_date": None, "label_realized_at": None, "label_valid": False, "invalid_reason": None, "target_price_basis": "pre_adjusted_close:收盘价前复权", "source_file_sha256": source_hashes[code], "label_protocol_id": LABEL_PROTOCOL}
            if before.empty: row["invalid_reason"] = "NO_TRADE_STRICTLY_BEFORE_ORIGIN"
            elif len(after) < 5: row["invalid_reason"] = "FEWER_THAN_FOUR_TRADES_AFTER_ORIGIN"
            else:
                anchor, target = before.iloc[-1], after.iloc[4]
                row.update({"anchor_trade_date": anchor.trade_date.date().isoformat(), "first_trade_on_or_after_origin": after.iloc[0].trade_date.date().isoformat(), "target_trade_date": target.trade_date.date().isoformat(), "label_realized_at": target.trade_date.date().isoformat()})
                if not np.isfinite(anchor.close_qfq) or not np.isfinite(target.close_qfq) or anchor.close_qfq <= 0: row["invalid_reason"] = "INVALID_ADJUSTED_CLOSE"
                else: row.update({"h4_return": float(target.close_qfq / anchor.close_qfq - 1.0), "label_valid": True})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    for argument in ("--delivery-attestation", "--daily-root", "--daily-manifest", "--fundamental-events", "--universe", "--output-root", "--materialization-date"):
        parser.add_argument(argument, type=Path, required=True)
    parser.add_argument("--daily-snapshot-mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()
    if args.output_root.exists(): raise FileExistsError("FAIL_CLOSED_V4_OUTPUT_ALREADY_EXISTS")
    try: materialization_date = date.fromisoformat(str(args.materialization_date))
    except ValueError as error: raise ValueError("FAIL_CLOSED_V4_MATERIALIZATION_DATE") from error
    attestation = load_delivery_attestation(args.delivery_attestation, args.daily_manifest, args.fundamental_events, args.universe, materialization_date)
    codes = load_universe(args.universe); source_hashes = load_daily_manifest(args.daily_manifest, codes)
    daily_by_code = {code: load_daily(args.daily_root / f"{code}.csv", source_hashes[code]) for code in codes}
    for code, daily in daily_by_code.items(): assert_coverage(daily, code)
    args.output_root.mkdir(parents=True); source_root = args.output_root / "daily_source"; source_root.mkdir()
    numeric = np.empty((len(ORIGINS), 8, len(codes), len(NUMERIC_FIELDS)), dtype=np.float32); tech_rows: list[dict[str, object]] = []
    for stock_index, code in enumerate(codes):
        numeric[:, :, stock_index, :] = numeric_features(daily_by_code[code])["numeric"]
        source, destination = args.daily_root / f"{code}.csv", source_root / f"{code}.csv"
        if args.daily_snapshot_mode == "copy":
            shutil.copy2(source, destination)
        else:
            destination.hardlink_to(source)
        for origin in ORIGINS: tech_rows.append({"origin_date": origin.date().isoformat(), "stock_code": code, **technical_row(daily_by_code[code], origin)})
    np.savez_compressed(args.output_root / "FRESH_NUMERIC.npz", x=numeric, origin_dates=np.asarray(ORIGIN_TEXT), stock_codes=np.asarray(codes))
    pd.DataFrame(tech_rows, columns=["origin_date", "stock_code", *TECH_FIELDS]).to_parquet(args.output_root / "FRESH_TECHNICAL.parquet", index=False)
    fundamental_panel(pd.read_parquet(args.fundamental_events), codes).to_parquet(args.output_root / "FRESH_FUNDAMENTALS.parquet", index=False)
    universe_frame = pd.DataFrame({"origin_date": np.repeat(ORIGIN_TEXT, len(codes)), "stock_code": np.tile(codes, len(ORIGINS))})
    universe_frame.to_parquet(args.output_root / "FRESH_UNIVERSE.parquet", index=False)
    pd.DataFrame({"stock_code": codes, "path": [str((source_root / f"{code}.csv").resolve()) for code in codes], "sha256": [source_hashes[code] for code in codes]}).to_csv(source_root / "FROZEN_DAILY_SOURCE_MANIFEST.csv", index=False, encoding="utf-8-sig")
    labels = sealed_labels(daily_by_code, source_hashes, codes)
    if not labels.label_valid.all(): raise RuntimeError("FAIL_CLOSED_V4_INVALID_LABEL_KEY")
    labels.to_parquet(args.output_root / "SEALED_FRESH_H4_LABELS.parquet", index=False)
    hashes = {name: sha256(args.output_root / name) for name in ("FRESH_NUMERIC.npz", "FRESH_TECHNICAL.parquet", "FRESH_FUNDAMENTALS.parquet", "FRESH_UNIVERSE.parquet", "SEALED_FRESH_H4_LABELS.parquet")}
    hashes["daily_source_manifest"] = sha256(source_root / "FROZEN_DAILY_SOURCE_MANIFEST.csv")
    receipt = {"node_id": "AA_GFMNET_CSN_FUTURE_ANCHOR_ELIGIBLE_FRESH_V4_SEALED_INPUT_MATERIALIZATION_V1", "status": "PASS_V4_SEALED_INPUT_MATERIALIZATION", "delivery_id": attestation["delivery_id"], "origin_dates": list(ORIGIN_TEXT), "origin_semantics": "pre_registered_monday_0930_strict_pit", "stock_count": len(codes), "numeric_shape": list(numeric.shape), "label_protocol_id": LABEL_PROTOCOL, "labels_read": False, "labels_opened_by_materialization": False, "label_rows_disclosed": False, "returns_read": False, "fresh_labels_read": False, "production_kernel_modified": False, "gpu_jobs_concurrent": 0, "cpu_thread_cap": 1, "output_sha256": hashes, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.output_root / "MATERIALIZATION_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("status", "origin_dates", "stock_count", "numeric_shape", "labels_read", "production_kernel_modified", "gpu_jobs_concurrent", "cpu_thread_cap")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

