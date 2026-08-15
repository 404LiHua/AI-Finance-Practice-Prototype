"""Build reproducible expanded panel_v2 batches from the locally archived A-share data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.custody import StageEDataCustodyGuard  # noqa: E402
from stage_e.hashing import (  # noqa: E402
    canonical_row_set_sha256,
    manifest_root_sha256,
    sha256_file,
    stable_json_sha256,
)


DEVELOPMENT_CEILING = pd.Timestamp("2023-06-02")
DEFAULT_CUSTODY = REPO_ROOT / "stage_e/configs/data_custody_v1.json"
KEY_COLUMNS = ["trade_date", "stock_code"]

COL = {
    "code": "\u80a1\u7968\u4ee3\u7801",
    "name": "\u80a1\u7968\u540d\u79f0",
    "industry": "\u884c\u4e1a",
    "listing_date": "\u4e0a\u5e02\u65e5\u671f",
    "market_type": "\u5e02\u573a\u7c7b\u578b",
    "exchange": "\u4ea4\u6613\u6240\u4ee3\u7801",
    "enterprise_nature": "\u4f01\u4e1a\u6027\u8d28",
    "date": "\u4ea4\u6613\u65e5\u671f",
    "weekly_close": "\u5468\u6536\u76d8\u4ef7",
    "weekly_open": "\u5468\u5f00\u76d8\u4ef7",
    "weekly_high": "\u5468\u6700\u9ad8\u4ef7",
    "weekly_low": "\u5468\u6700\u4f4e\u4ef7",
    "weekly_volume": "\u5468\u6210\u4ea4\u91cf(\u624b)",
    "weekly_amount": "\u5468\u6210\u4ea4\u989d(\u5343\u5143)",
    "close": "\u6536\u76d8\u4ef7",
    "open": "\u5f00\u76d8\u4ef7",
    "high": "\u6700\u9ad8\u4ef7",
    "low": "\u6700\u4f4e\u4ef7",
    "volume": "\u6210\u4ea4\u91cf(\u624b)",
    "amount": "\u6210\u4ea4\u989d(\u5343\u5143)",
    "factor": "\u590d\u6743\u56e0\u5b50",
    "qfq_open": "\u5f00\u76d8\u4ef7\u524d\u590d\u6743",
    "qfq_close": "\u6536\u76d8\u4ef7\u524d\u590d\u6743",
    "qfq_high": "\u6700\u9ad8\u4ef7\u524d\u590d\u6743",
    "qfq_low": "\u6700\u4f4e\u4ef7\u524d\u590d\u6743",
}

INDUSTRY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("金融", ("银行", "保险", "证券", "多元金融", "金融")),
    ("房地产", ("地产", "房地产", "园区开发", "房产服务")),
    ("信息技术", ("软件", "计算机", "IT", "半导体", "元器件", "通信", "互联网", "电子")),
    ("医疗健康", ("医药", "医疗", "生物", "制药", "中成药", "化学药")),
    ("可选消费", ("汽车", "摩托车", "家电", "家用电器", "传媒", "影视", "音像", "出版", "广告", "旅游", "酒店", "服饰", "文教休闲")),
    ("日常消费", ("食品", "饮料", "酿酒", "白酒", "啤酒", "农业", "农林", "林业", "牧渔", "渔业", "商业", "百货", "商品城", "超市", "连锁", "零售")),
    ("工业", ("机械", "机床", "设备", "建筑", "工程", "装修", "装饰", "电气", "电器仪表", "仪器", "运输设备", "国防军工")),
    ("原材料", ("化工", "农药", "化肥", "化纤", "塑料", "橡胶", "金属", "有色", "钢铁", "普钢", "铅锌", "铝", "铜", "建材", "玻璃", "水泥", "陶瓷", "造纸", "纺织", "包装")),
    ("能源", ("煤炭", "石油", "石化", "天然气", "油气")),
    ("公用事业", ("电力", "发电", "供水", "水务", "燃气", "供气", "供热", "环保", "环境保护", "公用事业")),
    ("交通运输", ("交通", "运输", "水运", "港口", "机场", "航空", "铁路", "路桥", "物流", "仓储")),
    ("综合", ("综合",)),
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_code(value: Any) -> str:
    text = re.sub(r"\.0$", "", str(value).strip())
    if "." in text:
        return text
    six = text.zfill(6)
    exchange = "SH" if six.startswith(("5", "6", "9")) else "SZ"
    return f"{six}.{exchange}"


def broad_industry(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    folded = text.casefold()
    for group, keywords in INDUSTRY_RULES:
        if any(keyword.casefold() in folded for keyword in keywords):
            return group
    return "其他"


def week_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("W-FRI").dt.end_time.dt.normalize()


def read_zip_table(root: Path, pattern: str, member: str) -> tuple[pd.DataFrame, Path]:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {pattern}, found {len(matches)}")
    path = matches[0]
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_excel(io.BytesIO(archive.read(member)), dtype=str)
    frame["source_row_number"] = np.arange(2, len(frame) + 2, dtype=np.int64)
    frame["stock_code"] = frame["Stkcd"].map(normalize_code)
    frame["source_file"] = path.name
    frame["source_sha256"] = sha256_file(path)
    return frame, path


def load_reference_data(data_root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[Path]]:
    basic_path = data_root / "stock_basic" / "stock_basic.csv"
    basic = pd.read_csv(basic_path, dtype={COL["code"]: str})
    keep = [COL[key] for key in ("code", "name", "industry", "listing_date", "market_type", "exchange", "enterprise_nature")]
    basic = basic[keep].rename(columns={
        COL["code"]: "stock_code", COL["name"]: "stock_name", COL["industry"]: "industry_snapshot",
        COL["listing_date"]: "listing_date", COL["market_type"]: "market_type",
        COL["exchange"]: "exchange_code", COL["enterprise_nature"]: "enterprise_nature_snapshot",
    })
    basic["stock_code"] = basic["stock_code"].map(normalize_code)
    basic["listing_date"] = pd.to_datetime(basic["listing_date"], errors="coerce")
    basic["industry_group"] = basic["industry_snapshot"].map(broad_industry)
    if basic["stock_code"].duplicated().any():
        raise ValueError("stock_basic has duplicate stock codes")

    csmar_root = data_root / "csmar"
    capital, capital_zip = read_zip_table(csmar_root, "*\u80a1\u672c\u7ed3\u6784*.zip", "SPT_Capchg.xlsx")
    special, special_zip = read_zip_table(csmar_root, "*\u7279\u6b8a\u5904\u7406*.zip", "SPT_Trdchg.xlsx")
    listing, listing_zip = read_zip_table(csmar_root, "*\u4e0a\u5e02\u72b6\u6001*.zip", "SPT_LTDSTACHG.xlsx")
    company, company_zip = read_zip_table(csmar_root, "*\u516c\u53f8\u6587\u4ef6*.zip", "SPT_Company.xlsx")
    for frame, columns in (
        (capital, ("Shrchgdt",)),
        (special, ("Annoudt", "Execudt")),
        (listing, ("Annoudt", "Execudt")),
        (company, ("Listdt", "Statdt")),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    capital["total_shares"] = pd.to_numeric(capital["Nshrttl"], errors="coerce")
    capital["tradable_a_shares"] = pd.to_numeric(capital["Nshra"], errors="coerce")
    capital["capital_effective_date"] = capital["Shrchgdt"]
    special["special_observable_at"] = special[["Annoudt", "Execudt"]].max(axis=1)
    listing["listing_status_observable_at"] = listing[["Annoudt", "Execudt"]].max(axis=1)
    return basic, {"capital": capital, "special": special, "listing": listing, "company": company}, [
        basic_path, capital_zip, special_zip, listing_zip, company_zip,
    ]


def scan_candidates(
    data_root: Path,
    basic: pd.DataFrame,
    events: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    minimum_coverage: float,
) -> pd.DataFrame:
    factor_codes = {path.stem for path in (data_root / "stk_factor").glob("*.csv")}
    expected_weeks = len(pd.date_range(start, end, freq="W-FRI"))
    records: list[dict[str, Any]] = []
    usecols = [COL["date"], COL["weekly_close"], COL["weekly_volume"]]
    for path in sorted((data_root / "weekly").glob("*.csv")):
        if path.stem not in factor_codes:
            continue
        frame = pd.read_csv(path, usecols=usecols)
        dates = pd.to_datetime(frame[COL["date"]], errors="coerce")
        mask = dates.between(start, end)
        if not mask.any():
            continue
        selected = frame.loc[mask].copy()
        selected_dates = dates.loc[mask]
        selected["week"] = week_end(selected_dates)
        selected["close"] = pd.to_numeric(selected[COL["weekly_close"]], errors="coerce")
        selected["volume"] = pd.to_numeric(selected[COL["weekly_volume"]], errors="coerce")
        selected = selected.dropna(subset=["week"]).sort_values("week").drop_duplicates("week", keep="last")
        records.append({
            "stock_code": path.stem,
            "observed_weeks": int(selected["week"].nunique()),
            "coverage_ratio": float(selected["week"].nunique() / expected_weeks),
            "last_observation": selected_dates.max(),
            "cutoff_close_unadjusted": selected["close"].iloc[-1] if not selected.empty else np.nan,
            "nonzero_volume_weeks": int(selected["volume"].gt(0).sum()),
        })
    candidates = basic.merge(pd.DataFrame(records), on="stock_code", how="inner", validate="one_to_one")

    capital = events["capital"].loc[
        events["capital"]["capital_effective_date"].le(end)
    ].sort_values(["stock_code", "capital_effective_date"]).groupby("stock_code", as_index=False).tail(1)
    candidates = candidates.merge(
        capital[["stock_code", "capital_effective_date", "total_shares", "tradable_a_shares"]],
        on="stock_code", how="left", validate="one_to_one",
    )
    candidates["market_cap_total_cutoff"] = candidates["cutoff_close_unadjusted"] * candidates["total_shares"]
    candidates["market_cap_float_cutoff"] = candidates["cutoff_close_unadjusted"] * candidates["tradable_a_shares"]
    candidates = candidates[
        candidates["listing_date"].le(start)
        & candidates["coverage_ratio"].ge(minimum_coverage)
        & candidates["last_observation"].ge(end - pd.Timedelta(days=14))
        & candidates["market_cap_total_cutoff"].gt(0)
    ].copy()
    if len(candidates) < 3:
        raise ValueError("fewer than three eligible candidates with point-in-time market cap")
    rank = candidates["market_cap_total_cutoff"].rank(method="first", pct=True)
    candidates["market_cap_bucket_cutoff"] = pd.cut(
        rank, bins=[0, 1 / 3, 2 / 3, 1], labels=["small", "mid", "large"], include_lowest=True
    ).astype(str)
    return candidates.sort_values("stock_code").reset_index(drop=True)


def select_stratified(candidates: pd.DataFrame, target: int) -> pd.DataFrame:
    if len(candidates) < target:
        raise ValueError(f"only {len(candidates)} candidates are eligible; target is {target}")
    work = candidates.copy()
    work["selection_priority"] = work["coverage_ratio"].rank(method="dense", ascending=False)
    cells = {
        (industry, bucket): group.sort_values(["selection_priority", "stock_code"]).to_dict("records")
        for (industry, bucket), group in work.groupby(["industry_group", "market_cap_bucket_cutoff"])
    }
    ordered_cells = sorted(cells, key=lambda key: (key[1], key[0]))
    selected: list[dict[str, Any]] = []
    while len(selected) < target:
        progressed = False
        for key in ordered_cells:
            if cells[key] and len(selected) < target:
                selected.append(cells[key].pop(0))
                progressed = True
        if not progressed:
            break
    result = pd.DataFrame(selected).drop_duplicates("stock_code").head(target).copy()
    if len(result) != target:
        raise ValueError(f"stratified selection produced {len(result)} stocks, expected {target}")
    result.insert(0, "selection_rank", np.arange(1, len(result) + 1))
    return result


def load_factor_week(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [
        COL["date"], COL["open"], COL["high"], COL["low"], COL["close"], COL["volume"], COL["amount"],
        COL["factor"], COL["qfq_open"], COL["qfq_high"], COL["qfq_low"], COL["qfq_close"],
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame["observation_trade_date"] = pd.to_datetime(frame[COL["date"]], errors="coerce")
    frame = frame[frame["observation_trade_date"].between(start, end)].copy()
    frame["trade_date"] = week_end(frame["observation_trade_date"])
    numeric = [column for column in columns if column != COL["date"]]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("observation_trade_date")
    weekly = frame.groupby("trade_date", as_index=False).agg(
        observation_trade_date=("observation_trade_date", "last"),
        open_unadjusted=(COL["open"], "first"), high_unadjusted=(COL["high"], "max"),
        low_unadjusted=(COL["low"], "min"), close_unadjusted=(COL["close"], "last"),
        model_open=(COL["qfq_open"], "first"), model_high=(COL["qfq_high"], "max"),
        model_low=(COL["qfq_low"], "min"), model_close=(COL["qfq_close"], "last"),
        model_volume_hands=(COL["volume"], "sum"), model_amount_thousand_cny=(COL["amount"], "sum"),
        adjust_factor_asof=(COL["factor"], "last"), trading_days_observed=("observation_trade_date", "size"),
    )
    return weekly


def load_raw_week(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [COL["date"], COL["weekly_open"], COL["weekly_high"], COL["weekly_low"], COL["weekly_close"], COL["weekly_volume"], COL["weekly_amount"]]
    frame = pd.read_csv(path, usecols=columns)
    frame["raw_week_observation_date"] = pd.to_datetime(frame[COL["date"]], errors="coerce")
    frame = frame[frame["raw_week_observation_date"].between(start, end)].copy()
    frame["trade_date"] = week_end(frame["raw_week_observation_date"])
    rename = {
        COL["weekly_open"]: "raw_week_open", COL["weekly_high"]: "raw_week_high",
        COL["weekly_low"]: "raw_week_low", COL["weekly_close"]: "raw_week_close",
        COL["weekly_volume"]: "raw_week_volume_hands", COL["weekly_amount"]: "raw_week_amount_thousand_cny",
    }
    frame = frame.rename(columns=rename)
    for column in rename.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["trade_date", "raw_week_observation_date", *rename.values()]].sort_values("raw_week_observation_date").drop_duplicates("trade_date", keep="last")


def asof_join(panel: pd.DataFrame, events: pd.DataFrame, date_column: str, columns: Iterable[str]) -> pd.DataFrame:
    result: list[pd.DataFrame] = []
    selected_columns = [date_column, *columns]
    for code, left in panel.groupby("stock_code", sort=False):
        right = events.loc[events["stock_code"].eq(code), selected_columns].dropna(subset=[date_column]).sort_values(date_column)
        left = left.sort_values("trade_date")
        if right.empty:
            joined = left.copy()
            for column in selected_columns:
                joined[column] = pd.NaT if column == date_column else np.nan
        else:
            joined = pd.merge_asof(left, right, left_on="trade_date", right_on=date_column, direction="backward")
        result.append(joined)
    return pd.concat(result, ignore_index=True).sort_values(KEY_COLUMNS).reset_index(drop=True)


def aggregate_text(events: dict[str, pd.DataFrame], selected_codes: set[str], end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    special = events["special"]
    special = special[special["stock_code"].isin(selected_codes) & special["special_observable_at"].le(end)]
    for row in special.itertuples(index=False):
        rows.append({
            "stock_code": row.stock_code, "published_at": row.special_observable_at,
            "text_title": f"special treatment change: {getattr(row, 'Stknmebc', '')} -> {getattr(row, 'Stknmeac', '')}",
            "text_body": f"change_type={getattr(row, 'Chgtype', '')}; reason={getattr(row, 'Chgrsdis', '')}",
            "text_source": "CSMAR SPT_Trdchg", "text_source_sha256": row.source_sha256,
        })
    capital = events["capital"]
    capital = capital[capital["stock_code"].isin(selected_codes) & capital["capital_effective_date"].le(end)]
    for row in capital.itertuples(index=False):
        rows.append({
            "stock_code": row.stock_code, "published_at": row.capital_effective_date,
            "text_title": "capital structure change",
            "text_body": f"total_shares={row.total_shares}; tradable_a_shares={row.tradable_a_shares}",
            "text_source": "CSMAR SPT_Capchg", "text_source_sha256": row.source_sha256,
        })
    if not rows:
        return pd.DataFrame(columns=["stock_code", "trade_date", "text_count", "text_title", "text_body", "text_sources", "text_source_hashes", "text_available"])
    text = pd.DataFrame(rows)
    text["trade_date"] = week_end(text["published_at"])
    return text.groupby(["stock_code", "trade_date"], as_index=False).agg(
        text_count=("text_title", "size"),
        text_title=("text_title", lambda values: " [SEP] ".join(map(str, values))),
        text_body=("text_body", lambda values: " [SEP] ".join(map(str, values))),
        text_sources=("text_source", lambda values: "|".join(sorted(set(map(str, values))))),
        text_source_hashes=("text_source_sha256", lambda values: "|".join(sorted(set(map(str, values))))),
    ).assign(text_available=True)


def build_panel(data_root: Path, selection: pd.DataFrame, events: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp, minimum_cross_section_coverage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(start, end, freq="W-FRI")
    codes = selection["stock_code"].tolist()
    skeleton = pd.MultiIndex.from_product([dates, codes], names=KEY_COLUMNS).to_frame(index=False)
    observed_frames = []
    for code in codes:
        factor = load_factor_week(data_root / "stk_factor" / f"{code}.csv", start, end)
        raw = load_raw_week(data_root / "weekly" / f"{code}.csv", start, end)
        merged = factor.merge(raw, on="trade_date", how="outer", validate="one_to_one")
        merged["stock_code"] = code
        observed_frames.append(merged)
    observed = pd.concat(observed_frames, ignore_index=True)
    panel = skeleton.merge(observed, on=KEY_COLUMNS, how="left", validate="one_to_one")
    static_columns = ["stock_code", "stock_name", "industry_snapshot", "industry_group", "listing_date", "market_type", "exchange_code", "enterprise_nature_snapshot", "market_cap_bucket_cutoff"]
    panel = panel.merge(selection[static_columns], on="stock_code", how="left", validate="many_to_one")

    listing = events["listing"].copy()
    panel = asof_join(panel, listing, "listing_status_observable_at", ["Stkstatac", "Chgtype", "source_sha256", "source_row_number"])
    panel = panel.rename(columns={"Stkstatac": "listing_status_after", "Chgtype": "listing_status_change_type", "source_sha256": "listing_status_source_sha256", "source_row_number": "listing_status_source_row"})
    panel["is_listed_asof"] = panel["trade_date"].ge(panel["listing_date"])
    status = panel["listing_status_after"].fillna("").astype(str)
    panel["is_delisted_asof"] = status.str.contains("\u7ec8\u6b62|\u9000\u5e02|\u6458\u724c|delist", case=False, regex=True)
    panel["is_suspended_listing_asof"] = status.str.contains("\u6682\u505c|suspend", case=False, regex=True)
    panel["listing_status_asof"] = np.where(~panel["is_listed_asof"], "not_listed", np.where(status.eq(""), "normal_or_unknown", status))
    panel["universe_member_pit"] = panel["is_listed_asof"] & ~panel["is_delisted_asof"]

    capital = events["capital"]
    panel = asof_join(panel, capital, "capital_effective_date", ["total_shares", "tradable_a_shares", "Shrtyp", "source_sha256", "source_row_number"])
    panel = panel.rename(columns={"capital_effective_date": "capital_effective_date_asof", "total_shares": "total_shares_asof", "tradable_a_shares": "tradable_a_shares_asof", "Shrtyp": "capital_change_type_asof", "source_sha256": "capital_source_sha256", "source_row_number": "capital_source_row"})
    special = events["special"]
    panel = asof_join(panel, special, "special_observable_at", ["Chgtype", "source_sha256", "source_row_number"])
    panel = panel.rename(columns={"special_observable_at": "special_status_observable_at", "Chgtype": "special_status_asof", "source_sha256": "special_status_source_sha256", "source_row_number": "special_status_source_row"})

    panel["has_price_observation"] = panel["model_close"].notna()
    panel["is_zero_volume_observation"] = panel["has_price_observation"] & panel["model_volume_hands"].fillna(0).le(0)
    panel["is_no_weekly_bar"] = panel["universe_member_pit"] & ~panel["has_price_observation"]
    panel["is_suspended"] = panel["universe_member_pit"] & (panel["is_suspended_listing_asof"] | panel["is_zero_volume_observation"])
    panel["is_tradable_pit"] = panel["universe_member_pit"] & panel["has_price_observation"] & ~panel["is_suspended"]
    panel["model_eligible_pit"] = panel["is_tradable_pit"]
    panel["is_special_treatment"] = ~panel["special_status_asof"].fillna("A").astype(str).str[-1:].isin(["", "A"])
    panel["trade_state"] = np.select(
        [~panel["is_listed_asof"], panel["is_delisted_asof"], panel["is_suspended"], panel["is_no_weekly_bar"], panel["is_tradable_pit"]],
        ["not_listed", "delisted", "suspended", "no_weekly_bar", "tradable"], default="inactive_or_unknown",
    )

    panel = panel.sort_values(KEY_COLUMNS).reset_index(drop=True)
    market_open = panel.groupby("trade_date")["has_price_observation"].any()
    panel["is_market_open_week"] = panel["trade_date"].map(market_open)
    grouped = panel.groupby("stock_code", sort=False)
    panel["history_weeks_available"] = grouped["has_price_observation"].cumsum()
    panel["return_1w"] = grouped["model_close"].pct_change(fill_method=None)
    panel["log_return_1w"] = grouped["model_close"].transform(lambda values: np.log(values).diff())
    for window in (4, 12, 26):
        panel[f"close_ma_{window}"] = grouped["model_close"].transform(lambda values, size=window: values.rolling(size, min_periods=size).mean())
        panel[f"return_vol_{window}"] = grouped["return_1w"].transform(lambda values, size=window: values.rolling(size, min_periods=size).std())
    panel["target_close"] = grouped["model_close"].transform(lambda values: values.shift(-1).bfill())
    observable_dates = panel["trade_date"].where(panel["has_price_observation"])
    panel["target_date"] = observable_dates.groupby(panel["stock_code"], sort=False).transform(
        lambda values: values.shift(-1).bfill()
    )
    panel["target_return"] = panel["target_close"] / panel["model_close"] - 1.0
    panel["target_direction"] = np.where(panel["target_return"].notna(), panel["target_return"].gt(0).astype(np.int8), np.nan)

    panel["market_cap_total_asof"] = panel["close_unadjusted"] * panel["total_shares_asof"]
    panel["market_cap_float_asof"] = panel["close_unadjusted"] * panel["tradable_a_shares_asof"]
    panel["market_cap_bucket_pit"] = "unknown"
    for date, indices in panel.groupby("trade_date").groups.items():
        values = panel.loc[indices, "market_cap_total_asof"]
        valid = values.dropna()
        if len(valid) >= 3:
            ranks = valid.rank(method="first", pct=True)
            panel.loc[valid.index, "market_cap_bucket_pit"] = pd.cut(ranks, [0, 1 / 3, 2 / 3, 1], labels=["small", "mid", "large"], include_lowest=True).astype(str)

    capital_week_keys = set(zip(capital["stock_code"], week_end(capital["capital_effective_date"])))
    keys = list(zip(panel["stock_code"], panel["trade_date"]))
    panel["capital_change_this_week"] = [key in capital_week_keys for key in keys]
    panel["adjust_factor_change_this_week"] = grouped["adjust_factor_asof"].transform(lambda values: values.ne(values.shift(1)) & values.shift(1).notna())
    panel["corporate_action_this_week"] = panel["capital_change_this_week"] | panel["adjust_factor_change_this_week"]
    panel["forward_adjust_factor_asof"] = panel["adjust_factor_asof"]
    panel["back_adjust_factor_asof"] = np.nan
    panel["adjust_factor_effective_date_asof"] = panel["trade_date"].where(panel["adjust_factor_change_this_week"]).groupby(panel["stock_code"]).ffill()

    text = aggregate_text(events, set(codes), end)
    panel = panel.merge(text, on=KEY_COLUMNS, how="left", validate="one_to_one")
    panel["text_count"] = panel["text_count"].fillna(0).astype(np.int32)
    panel["text_available"] = panel["text_available"].fillna(False).astype(bool)
    for column in ("text_title", "text_body", "text_sources", "text_source_hashes"):
        panel[column] = panel[column].fillna("")

    coverage = panel.groupby("trade_date")["model_eligible_pit"].mean().where(market_open)
    panel["cross_section_coverage"] = panel["trade_date"].map(coverage)
    panel["cross_section_eligible"] = panel["cross_section_coverage"].ge(minimum_cross_section_coverage)
    panel["sample_eligible_v2"] = (
        panel["model_eligible_pit"] & panel["cross_section_eligible"] & panel["target_return"].notna()
        & panel["target_date"].le(end) & panel["history_weeks_available"].ge(12)
    )
    panel["split"] = "development"
    panel["model_price_basis"] = "local_stk_factor_daily_qfq_aggregated_weekly"

    ledger_columns = ["stock_code", "capital_effective_date", "Shrtyp", "total_shares", "tradable_a_shares", "source_file", "source_sha256", "source_row_number"]
    ledger = capital[capital["stock_code"].isin(codes) & capital["capital_effective_date"].le(end)][ledger_columns].copy()
    return panel.sort_values(KEY_COLUMNS).reset_index(drop=True), ledger.sort_values(["stock_code", "capital_effective_date"])


def validate(panel: pd.DataFrame, selection: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    target = int(config["universe"]["target_stock_count"])
    start, end = pd.Timestamp(config["date_range"]["start"]), pd.Timestamp(config["date_range"]["end"])
    expected_dates = len(pd.date_range(start, end, freq="W-FRI"))
    industry_counts = selection["industry_group"].value_counts().sort_index().to_dict()
    cap_counts = selection["market_cap_bucket_cutoff"].value_counts().sort_index().to_dict()
    minimum_bucket = int(np.floor(target * float(config["universe"]["minimum_cap_bucket_fraction"])))
    open_panel = panel[panel["is_market_open_week"]]
    per_stock_open_coverage = open_panel.groupby("stock_code")["has_price_observation"].mean()
    open_cross_section = open_panel.groupby("trade_date")["model_eligible_pit"].mean()
    report = {
        "passed": True,
        "rows": int(len(panel)), "stocks": int(panel["stock_code"].nunique()), "dates": int(panel["trade_date"].nunique()),
        "expected_rows": target * expected_dates, "expected_dates": expected_dates,
        "duplicate_keys": int(panel.duplicated(KEY_COLUMNS).sum()),
        "missing_keys": int(panel[KEY_COLUMNS].isna().any(axis=1).sum()),
        "maximum_trade_date": panel["trade_date"].max().date().isoformat(),
        "future_trade_rows": int(panel["trade_date"].gt(DEVELOPMENT_CEILING).sum()),
        "market_open_weeks": int(panel.loc[panel["is_market_open_week"], "trade_date"].nunique()),
        "full_market_closed_weeks": int(panel.loc[~panel["is_market_open_week"], "trade_date"].nunique()),
        "minimum_per_stock_observation_coverage": float(per_stock_open_coverage.min()),
        "minimum_cross_section_coverage": float(open_cross_section.min()),
        "industry_group_count": int(selection["industry_group"].nunique()), "industry_counts": industry_counts,
        "market_cap_bucket_counts": cap_counts, "minimum_required_per_cap_bucket": minimum_bucket,
        "market_cap_known_rows": int(panel["market_cap_total_asof"].notna().sum()),
        "text_event_rows": int(panel["text_count"].gt(0).sum()),
        "corporate_action_rows": int(panel["corporate_action_this_week"].sum()),
        "sample_eligible_rows": int(panel["sample_eligible_v2"].sum()),
        "failures": [],
    }
    assertions = {
        "stock_count": report["stocks"] == target,
        "dense_row_count": report["rows"] == report["expected_rows"],
        "date_count": report["dates"] == expected_dates,
        "unique_non_null_key": report["duplicate_keys"] == 0 and report["missing_keys"] == 0,
        "development_ceiling": report["future_trade_rows"] == 0 and report["maximum_trade_date"] == end.date().isoformat(),
        "per_stock_coverage": report["minimum_per_stock_observation_coverage"] >= float(config["universe"]["minimum_history_coverage"]),
        "open_week_cross_section_coverage": report["minimum_cross_section_coverage"] >= float(config["task"]["minimum_cross_section_coverage"]),
        "industry_coverage": report["industry_group_count"] >= int(config["universe"]["minimum_industry_groups"]),
        "market_cap_coverage": all(cap_counts.get(bucket, 0) >= minimum_bucket for bucket in ("small", "mid", "large")),
        "eligible_samples": report["sample_eligible_rows"] > 0,
    }
    report["assertions"] = assertions
    report["failures"] = [name for name, passed in assertions.items() if not passed]
    report["passed"] = not report["failures"]
    return report


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", compression={"method": "gzip", "mtime": 0})


def build(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    start, end = pd.Timestamp(config["date_range"]["start"]), pd.Timestamp(config["date_range"]["end"])
    if end != DEVELOPMENT_CEILING:
        raise ValueError(f"expanded development batches must end at {DEVELOPMENT_CEILING.date()}")
    data_root = resolve_path(config["paths"]["data_root"])
    output_root = resolve_path(config["paths"]["output_root"])
    guard = StageEDataCustodyGuard.from_config(DEFAULT_CUSTODY, REPO_ROOT)
    guard.assert_paths_allowed([data_root, output_root, config_path], purpose="expanded panel build")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    basic, events, reference_paths = load_reference_data(data_root)
    candidates = scan_candidates(data_root, basic, events, start, end, float(config["universe"]["minimum_history_coverage"]))
    selection = select_stratified(candidates, int(config["universe"]["target_stock_count"]))
    panel, ledger = build_panel(data_root, selection, events, start, end, float(config["task"]["minimum_cross_section_coverage"]))
    guard.assert_development_frame(panel)
    report = validate(panel, selection, config)
    if not report["passed"]:
        raise ValueError(f"expanded panel validation failed: {report['failures']}")

    panel_path = output_root / "panel_v2.csv.gz"
    selection_path = output_root / "selected_universe.csv"
    write_csv_gz(panel, panel_path)
    write_csv_gz(panel[KEY_COLUMNS + ["universe_member_pit", "is_tradable_pit", "model_eligible_pit", "sample_eligible_v2", "trade_state", "industry_group", "market_cap_bucket_pit"]], output_root / "universe_membership.csv.gz")
    write_csv_gz(ledger, output_root / "corporate_actions.csv.gz")
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")

    selected_source_paths = reference_paths + [
        data_root / folder / f"{code}.csv" for code in selection["stock_code"] for folder in ("weekly", "stk_factor")
    ]
    source_records = []
    for path in sorted(selected_source_paths, key=lambda item: item.as_posix()):
        source_records.append({
            "source_id": "local_archived_input", "relative_path": path.relative_to(data_root).as_posix(),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
        })
    manifest_path = output_root / "source_file_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in source_records), encoding="utf-8")
    (output_root / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "data_batch_id": config["data_batch_id"], "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_date_ceiling": end.date().isoformat(), "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path), "panel_sha256": sha256_file(panel_path),
        "panel_row_set_sha256": canonical_row_set_sha256(panel),
        "selected_universe_sha256": sha256_file(selection_path),
        "source_manifest_root_sha256": manifest_root_sha256(source_records),
        "source_manifest_sha256": sha256_file(manifest_path), "source_file_count": len(source_records),
        "processing_batch_sha256": stable_json_sha256({
            "data_batch_id": config["data_batch_id"], "config_sha256": sha256_file(config_path),
            "panel_sha256": sha256_file(panel_path), "panel_row_set_sha256": canonical_row_set_sha256(panel),
            "source_manifest_root_sha256": manifest_root_sha256(source_records),
        }),
        "selection_policy": {
            "point_in_time_cutoff": end.date().isoformat(),
            "eligibility": "listed before window start, local weekly coverage threshold, recent observation at cutoff, point-in-time total shares available",
            "stratification": "deterministic round-robin across broad industry group and cutoff market-cap tercile",
            "industry_limitation": "industry_snapshot comes from stock_basic and is not a historical industry-membership series",
            "market_cap": "unadjusted close multiplied by latest total/tradable A shares effective no later than the sample date",
        },
        "price_policy": {
            "model_prices": "daily forward-adjusted OHLC from stk_factor aggregated to W-FRI",
            "audit_prices": "unadjusted local weekly bars and unadjusted daily close",
        },
        "validation": report, "sealed_data_read": False, "future_screening_or_final_read": False,
    }
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
