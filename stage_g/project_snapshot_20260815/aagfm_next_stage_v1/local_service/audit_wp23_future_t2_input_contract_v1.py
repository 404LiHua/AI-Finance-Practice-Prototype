from __future__ import annotations

"""CPU-only future-T2 input gate; it never opens target or label columns."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN = re.compile(r"(target|label|ordinal|return|fresh|screening|final|holdout)", re.I)
RG3 = ["momentum_20d", "momentum_60d", "momentum_120d", "realized_volatility_20d", "realized_volatility_60d", "downside_volatility_60d", "current_drawdown_60d", "rsi_14", "macd_scaled", "bollinger_position_20", "amihud_20d", "zero_volume_fraction_20d", "volume_ratio_20d_60d", "intraday_range_mean_20d"]
RG2 = ["capital_event_this_week", "capital_event_increase_flag", "capital_event_decrease_flag", "log_total_shares_change_at_event", "log_tradable_shares_change_at_event", "tradable_share_ratio_change_at_event", "capital_event_age_260_scaled", "capital_history_missing_flag", "market_tradable_fraction", "market_eligible_fraction", "market_small_cap_fraction", "industry_tradable_fraction", "industry_eligible_fraction", "log1p_industry_member_count", "graph_mean_absolute_change", "graph_intra_industry_weight_fraction", "graph_mean_nonself_out_degree_scaled", "graph_max_nonself_out_degree_scaled"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("origins", "universe", "rg3", "rg2", "scale", "protocol", "output_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name).resolve() for name in ("origins", "universe", "rg3", "rg2", "scale", "protocol")}; output = args.output_root.resolve()
    if output.exists(): raise RuntimeError(f"refusing to overwrite output: {output}")
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_FUTURE_T2_DATA_DELIVERY": raise RuntimeError("WP23 protocol is not frozen")
    failures: list[str] = []
    frames = {name: pd.read_csv(path, dtype={"stock_code": str}) for name, path in paths.items() if name != "protocol"}
    for name, frame in frames.items():
        forbidden = sorted(column for column in frame.columns if FORBIDDEN.search(str(column)))
        if forbidden: failures.append(f"{name}_forbidden_columns:{','.join(forbidden)}")
    origins, universe, rg3, rg2, scale = (frames[name] for name in ("origins", "universe", "rg3", "rg2", "scale"))
    required = {"origins": {"trade_date", "cutoff_at_utc", "cutoff_rule_id"}, "universe": {"trade_date", "stock_code", "eligible", "membership_effective_at"}, "rg3": {"trade_date", "stock_code", "source_trade_date", *RG3}, "rg2": {"trade_date", "stock_code", *RG2, "capital_effective_date_asof", "membership_state_date", "graph_state_date"}, "scale": {"trade_date", "stock_code", "market_volatility_4w"}}
    for name, cols in required.items():
        missing = sorted(cols.difference(frames[name].columns))
        if missing: failures.append(f"{name}_missing_columns:{','.join(missing)}")
    try:
        origins["trade_date"] = pd.to_datetime(origins["trade_date"], errors="raise").dt.normalize(); origins["cutoff_at_utc"] = pd.to_datetime(origins["cutoff_at_utc"], errors="raise", utc=True)
        for frame in (universe, rg3, rg2, scale): frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
        universe["membership_effective_at"] = pd.to_datetime(universe["membership_effective_at"], errors="raise", utc=True)
        for column in ("source_trade_date", "capital_effective_date_asof", "membership_state_date", "graph_state_date"):
            if column in rg3.columns: rg3[column] = pd.to_datetime(rg3[column], errors="raise").dt.normalize()
            if column in rg2.columns: rg2[column] = pd.to_datetime(rg2[column], errors="raise").dt.normalize()
    except (TypeError, ValueError, KeyError) as error:
        failures.append(f"timestamp_parse:{type(error).__name__}")
    origin_dates = set(origins.get("trade_date", pd.Series(dtype="datetime64[ns]")).dropna())
    if len(origin_dates) < int(protocol["minimum_qualified_origins"]): failures.append("fewer_than_12_origins")
    if origins.get("trade_date", pd.Series(dtype="datetime64[ns]")).duplicated().any(): failures.append("duplicate_origin")
    if len(origins) and not (origins.trade_date.dt.weekday == 4).all(): failures.append("origin_not_friday")
    if len(origins) and bool((origins.cutoff_at_utc.dt.tz_convert("Asia/Shanghai").dt.date != origins.trade_date.dt.date).any()): failures.append("cutoff_local_date_mismatch")
    if len(origins) and bool((origins.trade_date <= pd.Timestamp(protocol["development_cutoff_date"])).any()): failures.append("origin_not_after_development_cutoff")
    cutoff_by_date = dict(zip(origins.trade_date, origins.cutoff_at_utc))
    if len(universe) and bool((universe.membership_effective_at > universe.trade_date.map(cutoff_by_date)).fillna(True).any()): failures.append("membership_after_cutoff")
    if len(rg3) and bool((rg3.source_trade_date > rg3.trade_date).any()): failures.append("rg3_future_source_date")
    if len(rg2):
        for column in ("capital_effective_date_asof", "membership_state_date", "graph_state_date"):
            if column in rg2.columns and bool((rg2[column] > rg2.trade_date).any()): failures.append(f"rg2_future_pit_date:{column}")
    for name, frame in (("universe", universe), ("rg3", rg3), ("rg2", rg2), ("scale", scale)):
        if frame.duplicated(["trade_date", "stock_code"]).any(): failures.append(f"{name}_duplicate_key")
    eligible = universe[universe.eligible.astype(str).str.lower().isin({"true", "1", "yes"})][["trade_date", "stock_code"]].drop_duplicates()
    if len(eligible):
        counts = eligible.groupby("trade_date").size()
        if bool((counts < int(protocol["coverage"]["minimum_eligible_stocks_per_origin"])).any()): failures.append("eligible_stocks_below_300")
        eligible_keys = set(map(tuple, eligible.astype({"stock_code": str}).itertuples(index=False, name=None)))
        for name, frame in (("rg3", rg3), ("rg2", rg2), ("scale", scale)):
            keys = set(map(tuple, frame[["trade_date", "stock_code"]].astype({"stock_code": str}).itertuples(index=False, name=None)))
            if eligible_keys.difference(keys): failures.append(f"{name}_missing_eligible_keys:{len(eligible_keys.difference(keys))}")
    for name, frame in (("rg3", rg3), ("rg2", rg2), ("scale", scale)):
        numeric = frame[[column for column in frame.columns if column not in {"trade_date", "stock_code", "source_trade_date", "capital_effective_date_asof", "membership_state_date", "graph_state_date"}]].apply(pd.to_numeric, errors="coerce")
        if np.isinf(numeric.to_numpy(float)).any(): failures.append(f"{name}_infinite_value")
    result = {"node_id": "WP23_FUTURE_T2_INPUT_PRECONSUMPTION_AUDIT_V1", "status": "PASS_READY_FOR_PREDICTION_SEAL_AND_NEW_ONE_TIME_LABEL_AUTHORIZATION" if not failures else "FAIL_CLOSED_FUTURE_T2_INPUT_PRECONSUMPTION", "input_sha256": {name: sha256(path) for name, path in paths.items()}, "origin_count": int(len(origin_dates)), "eligible_rows": int(len(eligible)), "eligible_stock_count": int(eligible.stock_code.nunique()) if len(eligible) else 0, "failures": failures, "target_labels_read": False, "fresh_labels_read": False, "metrics_read": False, "model_trained": False, "gpu_used": False, "production_registry_modified": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True); (output / "WP23_FUTURE_T2_INPUT_PRECONSUMPTION_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": failures}, ensure_ascii=False))
    if failures: raise SystemExit(2)


if __name__ == "__main__": main()


