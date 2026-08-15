from __future__ import annotations

"""Build a label-free future RG2 state panel for the C1 shadow path.

The builder is deliberately metric-blind.  It reconstructs only the 18 PIT
state/graph-summary features from the frozen REV5.2 formula, using the
current CSMAR capital/status archives and the frozen 300-stock universe.  A
development-date replay against the archived ``NEW_INFORMATION_VIEW`` and
E-3 graph statistics is mandatory before a future panel is accepted.

No target labels, FRESH, SCREENING, FINAL, incumbent predictions, or model
registry are opened.  CPU is used for parsing and graph construction; GPU is
not touched.
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BAN = ("fresh", "screening", "final", "sealed_holdout")
FEATURES = [
    "capital_event_this_week", "capital_event_increase_flag", "capital_event_decrease_flag",
    "log_total_shares_change_at_event", "log_tradable_shares_change_at_event",
    "tradable_share_ratio_change_at_event", "capital_event_age_260_scaled",
    "capital_history_missing_flag", "market_tradable_fraction", "market_eligible_fraction",
    "market_small_cap_fraction", "industry_tradable_fraction", "industry_eligible_fraction",
    "log1p_industry_member_count", "graph_mean_absolute_change",
    "graph_intra_industry_weight_fraction", "graph_mean_nonself_out_degree_scaled",
    "graph_max_nonself_out_degree_scaled",
]
PIT_DATES = ["capital_effective_date_asof", "membership_state_date", "graph_state_date"]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_xlsx_table(zip_path: Path, workbook_suffix: str) -> list[dict[str, str]]:
    """Read a small CSMAR XLSX table without requiring openpyxl."""
    with zipfile.ZipFile(zip_path) as outer:
        names = [name for name in outer.namelist() if name.endswith(workbook_suffix + ".xlsx")]
        if len(names) != 1:
            raise RuntimeError(f"cannot identify {workbook_suffix}.xlsx in {zip_path}")
        payload = outer.read(names[0])
    with zipfile.ZipFile(io.BytesIO(payload)) as workbook:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.iter(NS + "t")) for item in shared_root.findall(NS + "si")]

        def cell_value(cell: ET.Element) -> str:
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                return "".join(node.text or "" for node in cell.iter(NS + "t"))
            value = cell.find(NS + "v")
            raw = "" if value is None else (value.text or "")
            return shared[int(raw)] if cell_type == "s" and raw else raw

        rows: list[dict[str, str]] = []
        headers: dict[str, str] | None = None
        with workbook.open("xl/worksheets/sheet1.xml") as stream:
            for row_index, (_, row) in enumerate(ET.iterparse(stream, events=("end",))):
                if row.tag != NS + "row":
                    continue
                values: dict[str, str] = {}
                for cell in row.findall(NS + "c"):
                    column = re.sub(r"\d", "", cell.attrib.get("r", ""))
                    values[column] = cell_value(cell)
                if headers is None:
                    headers = values
                elif values:
                    rows.append({headers.get(column, column): value for column, value in values.items()})
                row.clear()
        return rows


def read_daily(path: Path, code: str, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(path, usecols=[1, 15])
    raw.columns = ["trade_date", "close"]
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw[raw["trade_date"].notna() & raw["close"].notna()]
    raw = raw.sort_values("trade_date", kind="mergesort").drop_duplicates("trade_date", keep="last")
    raw = raw[raw["trade_date"] <= calendar.max()]
    weekly = (
        raw.assign(canonical_week=raw["trade_date"].dt.to_period("W-FRI").dt.end_time.dt.normalize())
        .groupby("canonical_week", as_index=True, sort=True)
        .tail(1)
        .set_index("canonical_week")
        .reindex(calendar)
    )
    return weekly[["close"]].rename(columns={"close": "model_close"}).reset_index(names="trade_date").assign(stock_code=code), sha256(path)


def row_normalize(adjacency: np.ndarray) -> np.ndarray:
    return adjacency / np.clip(adjacency.sum(axis=-1, keepdims=True), 1e-12, None)


def correlation_topk(correlation: np.ndarray, top_k: int = 8, minimum: float = 0.05, self_weight: float = 1.0) -> np.ndarray:
    count = correlation.shape[0]
    graph = np.zeros((count, count), dtype=np.float32)
    indices = np.arange(count)
    for row in range(count):
        scores = np.abs(correlation[row]).astype(float)
        scores[~np.isfinite(scores)] = -np.inf
        scores[row] = -np.inf
        order = np.lexsort((indices, -scores))
        selected = [index for index in order if scores[index] >= minimum][:top_k]
        graph[row, row] = self_weight
        if selected:
            graph[row, selected] = scores[selected]
    return row_normalize(graph).astype(np.float32)


def graph_stats(adjacency: np.ndarray, industries: list[str], previous: np.ndarray | None) -> dict[str, float]:
    nonself = adjacency.copy(); np.fill_diagonal(nonself, 0.0)
    edge_mask = nonself > 0
    same = np.equal.outer(np.asarray(industries, dtype=object), np.asarray(industries, dtype=object))
    total_weight = float(nonself.sum())
    return {
        "graph_mean_absolute_change": 0.0 if previous is None else float(np.abs(adjacency - previous).mean()),
        "graph_intra_industry_weight_fraction": float(nonself[same].sum() / total_weight) if total_weight else 1.0,
        "graph_mean_nonself_out_degree_scaled": float(edge_mask.sum(axis=1).mean()) / 299.0,
        "graph_max_nonself_out_degree_scaled": float(edge_mask.sum(axis=1).max(initial=0)) / 299.0,
    }


def load_capital_events(path: Path, selected: set[str]) -> dict[str, list[tuple[pd.Timestamp, float, float, str]]]:
    rows = parse_xlsx_table(path, "SPT_Capchg")
    events: dict[str, list[tuple[pd.Timestamp, float, float, str]]] = {code: [] for code in selected}
    for row in rows:
        code6 = str(row.get("Stkcd", "")).strip().zfill(6)
        code = next((item for item in selected if item.startswith(code6 + ".")), None)
        if code is None:
            continue
        date = pd.to_datetime(row.get("Shrchgdt", ""), errors="coerce")
        if pd.isna(date):
            continue
        def number(name: str) -> float:
            value = str(row.get(name, "")).strip()
            if not value:
                return float("nan")
            try:
                return float(value)
            except ValueError:
                return float("nan")
        events[code].append((pd.Timestamp(date).normalize(), number("Nshrttl"), number("Nshra"), str(row.get("Shrtyp", ""))))
    for code in events:
        events[code].sort(key=lambda item: item[0])
    return events


def load_status(path: Path, selected: set[str], target: pd.Timestamp) -> dict[str, str]:
    rows = parse_xlsx_table(path, "SPT_LTDSTACHG")
    status = {code: "A" for code in selected}
    latest: dict[str, tuple[pd.Timestamp, str]] = {}
    for row in rows:
        code6 = str(row.get("Stkcd", "")).strip().zfill(6)
        code = next((item for item in selected if item.startswith(code6 + ".")), None)
        if code is None:
            continue
        date = pd.to_datetime(row.get("Execudt") or row.get("Annoudt"), errors="coerce")
        if pd.isna(date) or pd.Timestamp(date).normalize() > target:
            continue
        transition = str(row.get("Chgtype", "")).strip()
        if not transition:
            continue
        candidate = (pd.Timestamp(date).normalize(), transition[-1])
        if code not in latest or candidate[0] >= latest[code][0]:
            latest[code] = candidate
    for code, (_, state) in latest.items():
        status[code] = state
    return status


def capital_vector(events: list[tuple[pd.Timestamp, float, float, str]], target: pd.Timestamp) -> dict[str, float | str]:
    available = [item for item in events if item[0] <= target]
    if not available:
        return {
            "capital_event_this_week": 0.0, "capital_event_increase_flag": 0.0, "capital_event_decrease_flag": 0.0,
            "log_total_shares_change_at_event": 0.0, "log_tradable_shares_change_at_event": 0.0,
            "tradable_share_ratio_change_at_event": 0.0, "capital_event_age_260_scaled": 1.0,
            "capital_history_missing_flag": 1.0, "capital_effective_date_asof": "",
        }
    latest = available[-1]; previous = available[-2] if len(available) > 1 else None
    this_week = latest[0] > target - pd.Timedelta(days=7)
    total_change = tradable_change = ratio_change = 0.0
    if this_week and previous is not None and all(np.isfinite(x) and x > 0 for x in (latest[1], latest[2], previous[1], previous[2])):
        total_change = math.log(latest[1] / previous[1]); tradable_change = math.log(latest[2] / previous[2])
        ratio_change = latest[2] / latest[1] - previous[2] / previous[1]
    return {
        "capital_event_this_week": float(this_week), "capital_event_increase_flag": float(this_week and total_change > 0),
        "capital_event_decrease_flag": float(this_week and total_change < 0),
        "log_total_shares_change_at_event": total_change, "log_tradable_shares_change_at_event": tradable_change,
        "tradable_share_ratio_change_at_event": ratio_change,
        "capital_event_age_260_scaled": min((target - latest[0]).days // 7, 260) / 260.0,
        "capital_history_missing_flag": 0.0, "capital_effective_date_asof": latest[0].date().isoformat(),
    }


def build_state_for_date(universe: pd.DataFrame, weekly: pd.DataFrame, events: dict[str, list[tuple[pd.Timestamp, float, float, str]]], status: dict[str, str], target: pd.Timestamp, graph: dict[str, float]) -> pd.DataFrame:
    frame = universe[["stock_code", "industry_group", "market_cap_bucket_cutoff"]].copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["industry_group"] = frame["industry_group"].fillna("其他").replace("", "其他")
    frame["market_cap_bucket_cutoff"] = frame["market_cap_bucket_cutoff"].fillna("mid")
    latest = weekly[weekly["trade_date"].eq(target)].set_index("stock_code")["model_close"]
    frame["observed"] = frame["stock_code"].map(latest).notna()
    frame["state_code_asof"] = frame["stock_code"].map(status).fillna("A")
    frame["tradable"] = frame["observed"] & frame["state_code_asof"].isin(["A", "T"])
    frame["eligible"] = frame["observed"] & frame["state_code_asof"].eq("A")
    market_n = float(len(frame))
    market_tradable = float(frame["tradable"].sum()) / market_n
    market_eligible = float(frame["eligible"].sum()) / market_n
    market_small = float(frame["market_cap_bucket_cutoff"].eq("small").sum()) / market_n
    group = frame.groupby("industry_group", sort=False)
    totals = group["stock_code"].transform("size").astype(float)
    tradable_counts = group["tradable"].transform("sum").astype(float)
    eligible_counts = group["eligible"].transform("sum").astype(float)
    result_rows = []
    for idx, row in frame.iterrows():
        cap = capital_vector(events.get(row.stock_code, []), target)
        result_rows.append({
            "trade_date": target, "stock_code": row.stock_code,
            **{key: cap[key] for key in cap if key in {"capital_event_this_week", "capital_event_increase_flag", "capital_event_decrease_flag", "log_total_shares_change_at_event", "log_tradable_shares_change_at_event", "tradable_share_ratio_change_at_event", "capital_event_age_260_scaled", "capital_history_missing_flag"}},
            "market_tradable_fraction": market_tradable, "market_eligible_fraction": market_eligible,
            "market_small_cap_fraction": market_small,
            "industry_tradable_fraction": float(tradable_counts.loc[idx] / totals.loc[idx]) if totals.loc[idx] else market_tradable,
            "industry_eligible_fraction": float(eligible_counts.loc[idx] / totals.loc[idx]) if totals.loc[idx] else market_eligible,
            "log1p_industry_member_count": float(math.log1p(totals.loc[idx])),
            **graph,
            "capital_effective_date_asof": cap["capital_effective_date_asof"],
            "membership_state_date": target.date().isoformat(), "graph_state_date": target.date().isoformat(),
            "state_code_asof": row.state_code_asof,
        })
    result = pd.DataFrame(result_rows)
    numeric = [column for column in FEATURES if column in result.columns]
    if result[numeric].isna().any().any() or not np.isfinite(result[numeric].to_numpy(float)).all():
        raise RuntimeError("future RG2 state contains missing/non-finite features")
    return result


def development_replay(state_source: Path, actions_path: Path, membership_path: Path, graph_path: Path, selected: pd.DataFrame) -> dict[str, object]:
    source = pd.read_csv(state_source, usecols=["trade_date", "stock_code", *FEATURES], dtype={"stock_code": str})
    target = pd.Timestamp("2023-05-05")
    expected = source[pd.to_datetime(source.trade_date).eq(target)].copy()
    if len(expected) != 300:
        raise RuntimeError(f"development state replay expected 300 rows, got {len(expected)}")
    actions = pd.read_csv(actions_path, dtype={"stock_code": str})
    actions["capital_effective_date"] = pd.to_datetime(actions["capital_effective_date"], errors="raise").dt.normalize()
    events: dict[str, list[tuple[pd.Timestamp, float, float, str]]] = {code: [] for code in selected.stock_code.astype(str)}
    for row in actions.itertuples(index=False):
        if row.stock_code in events:
            events[row.stock_code].append((row.capital_effective_date, float(row.total_shares), float(row.tradable_a_shares), str(row.Shrtyp)))
    for code in events: events[code].sort(key=lambda item: item[0])
    membership = pd.read_csv(membership_path, dtype={"stock_code": str})
    membership["trade_date"] = pd.to_datetime(membership.trade_date, errors="raise").dt.normalize()
    day = membership[membership.trade_date.eq(target)].copy()
    if len(day) != 300: raise RuntimeError("development membership replay coverage failure")
    # Reuse archived market/industry membership for the formula replay.
    selected_replay = selected[["stock_code", "industry_group", "market_cap_bucket_cutoff"]].copy()
    selected_replay["industry_group"] = day.set_index("stock_code").reindex(selected_replay.stock_code)["industry_group"].fillna("其他").to_numpy()
    selected_replay["market_cap_bucket_cutoff"] = day.set_index("stock_code").reindex(selected_replay.stock_code)["market_cap_bucket_pit"].fillna("mid").to_numpy()
    weekly = pd.DataFrame({"trade_date": [target] * 300, "stock_code": selected_replay.stock_code, "model_close": 1.0})
    status = {row.stock_code: ("A" if bool(row.is_tradable_pit) else "S") for row in day.itertuples(index=False)}
    graph_row = pd.read_csv(graph_path)
    graph_row["trade_date"] = pd.to_datetime(graph_row.trade_date, errors="raise").dt.normalize()
    g = graph_row[graph_row.trade_date.eq(target)].iloc[0]
    graph = {
        "graph_mean_absolute_change": float(g.mean_absolute_change),
        "graph_intra_industry_weight_fraction": float(g.intra_industry_weight_fraction),
        "graph_mean_nonself_out_degree_scaled": float(g.mean_nonself_out_degree) / 299.0,
        "graph_max_nonself_out_degree_scaled": float(g.maximum_nonself_out_degree) / 299.0,
    }
    built = build_state_for_date(selected_replay, weekly, events, status, target, graph)
    left = expected.sort_values("stock_code").reset_index(drop=True)
    right = built.sort_values("stock_code").reset_index(drop=True)
    errors = {}
    for column in FEATURES:
        if column not in right or not np.allclose(left[column].to_numpy(float), right[column].to_numpy(float), rtol=0.0, atol=1e-8, equal_nan=False):
            errors[column] = float(np.max(np.abs(left[column].to_numpy(float) - right[column].to_numpy(float))))
    if errors:
        raise RuntimeError(f"development RG2 formula replay failed: {errors}")
    return {"status": "PASS_DEVELOPMENT_RG2_FORMULA_REPLAY", "date": target.date().isoformat(), "rows": 300, "feature_count": len(FEATURES), "max_absolute_error": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-input", required=True, type=Path)
    parser.add_argument("--daily-root", required=True, type=Path)
    parser.add_argument("--selected-universe", required=True, type=Path)
    parser.add_argument("--capital-events-csv", required=True, type=Path)
    parser.add_argument("--capital-events-csmar-zip", required=True, type=Path)
    parser.add_argument("--status-csmar-zip", required=True, type=Path)
    parser.add_argument("--development-state-source", required=True, type=Path)
    parser.add_argument("--development-graph-stats", required=True, type=Path)
    parser.add_argument("--development-membership", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    shadow_path, daily_root, selected_path = args.shadow_input.resolve(), args.daily_root.resolve(), args.selected_universe.resolve()
    actions_path, cap_zip, status_zip = args.capital_events_csv.resolve(), args.capital_events_csmar_zip.resolve(), args.status_csmar_zip.resolve()
    dev_state, dev_graph, dev_membership, output = args.development_state_source.resolve(), args.development_graph_stats.resolve(), args.development_membership.resolve(), args.output_root.resolve()
    paths = (shadow_path, daily_root, selected_path, actions_path, cap_zip, status_zip, dev_state, dev_graph, dev_membership, output)
    if output.exists(): raise RuntimeError(f"refusing to overwrite output: {output}")
    if any(token in str(path).lower() for path in paths for token in BAN): raise RuntimeError("prohibited holdout/fresh path token")
    if not all(path.exists() for path in (shadow_path, daily_root, selected_path, actions_path, cap_zip, status_zip, dev_state, dev_graph, dev_membership)): raise RuntimeError("required RG2 future input missing")
    if not 1 <= args.workers <= 8: raise RuntimeError("workers must be between 1 and 8")
    shadow = pd.read_parquet(shadow_path)[["trade_date", "stock_code"]].copy()
    shadow["trade_date"] = pd.to_datetime(shadow.trade_date, errors="raise").dt.normalize(); shadow["stock_code"] = shadow.stock_code.astype(str)
    if len(shadow) != 300 or shadow.trade_date.nunique() != 1 or shadow.duplicated(["trade_date", "stock_code"]).any(): raise RuntimeError("future shadow key contract failure")
    target = pd.Timestamp(shadow.trade_date.iloc[0])
    universe = pd.read_csv(selected_path, dtype={"stock_code": str}).sort_values("selection_rank", kind="mergesort").reset_index(drop=True)
    if len(universe) != 300 or set(universe.stock_code.astype(str)) != set(shadow.stock_code): raise RuntimeError("future shadow and frozen universe mismatch")
    codes = universe.stock_code.astype(str).tolist(); code_set = set(codes)
    missing = [code for code in codes if not (daily_root / f"{code}.csv").is_file()]
    if missing: raise RuntimeError(f"missing daily source files: {missing[:10]}")
    calendar = pd.date_range("2018-06-08", target, freq="W-FRI")
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
    panels: list[pd.DataFrame] = []; daily_hashes: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(read_daily, daily_root / f"{code}.csv", code, calendar): code for code in codes}
        for future in as_completed(futures):
            code = futures[future]; panel, digest = future.result(); panels.append(panel); daily_hashes[code] = digest
    weekly = pd.concat(panels, ignore_index=True)
    weekly["trade_date"] = pd.to_datetime(weekly.trade_date, errors="raise").dt.normalize()
    pivot = weekly.pivot(index="trade_date", columns="stock_code", values="model_close").reindex(columns=codes)
    returns = pivot.pct_change(fill_method=None)
    position = returns.index.get_loc(target); window = returns.iloc[max(0, position - 26 + 1):position + 1]
    if len(window) < 12: raise RuntimeError("insufficient graph window")
    adjacency = correlation_topk(window.corr(min_periods=12).to_numpy(float))
    graph = graph_stats(adjacency, universe.industry_group.fillna("其他").astype(str).tolist(), None)
    previous_window = returns.iloc[max(0, position - 26):position]
    if len(previous_window) >= 12:
        previous = correlation_topk(previous_window.corr(min_periods=12).to_numpy(float))
        graph["graph_mean_absolute_change"] = float(np.abs(adjacency - previous).mean())
    events = load_capital_events(cap_zip, code_set)
    status = load_status(status_zip, code_set, target)
    # Mandatory formula replay before future materialization.
    replay = development_replay(dev_state, actions_path, dev_membership, dev_graph, universe)
    result = build_state_for_date(universe, weekly, events, status, target, graph)
    result = result.merge(shadow, on=["trade_date", "stock_code"], how="inner", validate="one_to_one", suffixes=("", "_shadow"))
    if len(result) != 300: raise RuntimeError("future RG2 key join incomplete")
    if any(bool((pd.to_datetime(result[column], errors="coerce") > target).any()) for column in PIT_DATES):
        raise RuntimeError("future PIT date violation")
    output.mkdir(parents=True)
    panel_path = output / "WP22_C1_FUTURE_RG2_STATE_FEATURES.parquet"; receipt_path = output / "WP22_C1_FUTURE_RG2_STATE_RECEIPT.json"
    result.to_parquet(panel_path, index=False, engine="pyarrow", compression="zstd")
    receipt = {
        "node_id": "WP22_C1_FUTURE_RG2_STATE_FEATURES_V1", "status": "PASS_LABEL_FREE_FUTURE_RG2_STATE_MATERIALIZED",
        "target_id": "T2_MARKET_RELATIVE_FIXED", "origin_date": target.date().isoformat(), "rows": 300,
        "feature_count": len(FEATURES), "features": FEATURES, "graph_formula": "E3 rolling Pearson correlation, 26-week window, minimum_periods=12, top_k=8, abs_corr>=0.05, self_weight=1",
        "membership_rule": "frozen 300 universe; weekly close observed at origin; state A=eligible, T=tradable-only, S/X=not tradable",
        "capital_rule": "latest CSMAR effective event at or before origin; this-week means latest event strictly newer than origin-7 days; age capped at 260 weeks",
        "input_sha256": {"shadow_input": sha256(shadow_path), "selected_universe": sha256(selected_path), "capital_events_csv": sha256(actions_path), "capital_events_csmar_zip": sha256(cap_zip), "status_csmar_zip": sha256(status_zip), "development_state_source": sha256(dev_state), "development_graph_stats": sha256(dev_graph), "development_membership": sha256(dev_membership), "daily_sources": daily_hashes},
        "output_sha256": sha256(panel_path), "development_replay": replay,
        "status_code_counts": pd.Series(status).value_counts().to_dict(),
        "pit_dates_checked": PIT_DATES, "target_labels_read": False, "fresh_labels_read": False, "screening_read": False, "final_read": False, "production_registry_modified": False, "gpu_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "origin_date": receipt["origin_date"], "rows": receipt["rows"], "output": str(panel_path), "output_sha256": receipt["output_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
