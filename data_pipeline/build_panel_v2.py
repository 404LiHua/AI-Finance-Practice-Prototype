"""Build a dense point-in-time panel_v2 over trade_date x stock_code x feature."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTRACT_VERSION = "panel_v2.0.0"
KEY_COLUMNS = ["trade_date", "stock_code"]
STATE_COLUMNS = [
    "listing_date",
    "is_listed_asof",
    "listing_status_asof",
    "listing_status_observable_at",
    "is_delisted_asof",
    "is_suspended_listing_asof",
    "universe_member_pit",
    "has_price_observation",
    "is_zero_volume_observation",
    "is_suspended",
    "is_no_weekly_bar",
    "is_tradable_pit",
    "model_eligible_pit",
    "sample_eligible_v2",
    "trade_state",
    "is_special_treatment",
    "total_shares_asof",
    "tradable_a_shares_asof",
    "capital_effective_date_asof",
    "capital_change_this_week",
    "forward_adjust_factor_asof",
    "back_adjust_factor_asof",
    "adjust_factor_asof",
    "adjust_factor_effective_date_asof",
    "adjust_factor_change_this_week",
    "corporate_action_this_week",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"contract_version must be {CONTRACT_VERSION}")
    return config


def resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_base_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"stock_code", "trade_date", "calendar_week_end"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"base panel missing columns: {missing}")
    result = frame.copy()
    result["observation_trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    result["trade_date"] = pd.to_datetime(result["calendar_week_end"], errors="coerce")
    result = result.drop(columns=["calendar_week_end"])
    result["stock_code"] = result["stock_code"].astype(str)
    if result[KEY_COLUMNS].isna().any().any():
        raise ValueError("base panel contains missing trade_date or stock_code")
    if result.duplicated(KEY_COLUMNS).any():
        raise ValueError("base panel contains duplicate trade_date/stock_code rows")
    return result


def normalize_stock_basic(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["stock_code", "listing_date"])
    mapping = {
        "股票代码": "stock_code",
        "股票名称": "stock_name",
        "行业": "industry",
        "上市状态": "listing_status_snapshot",
        "上市日期": "listing_date",
        "市场类型": "market_type",
        "交易所代码": "exchange_code",
    }
    result = frame.rename(columns={key: value for key, value in mapping.items() if key in frame.columns})
    required = {"stock_code", "listing_date"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"stock basic missing columns: {missing}")
    keep = [column for column in mapping.values() if column in result.columns]
    result = result[keep].copy()
    result["stock_code"] = result["stock_code"].astype(str)
    result["listing_date"] = pd.to_datetime(result["listing_date"], errors="coerce")
    if result["stock_code"].duplicated().any():
        raise ValueError("stock basic contains duplicate stock_code rows")
    return result


def normalize_company_records(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["stock_code", "company_listing_date"])
    result = frame.copy()
    result["stock_code"] = result["stock_code"].astype(str)
    result["company_listing_date"] = pd.to_datetime(result.get("Listdt"), errors="coerce")
    columns = ["stock_code", "company_listing_date"]
    for source, target in (("Nindcd", "industry_code_csmar"), ("Nindnme", "industry_name_csmar")):
        if source in result:
            result[target] = result[source]
            columns.append(target)
    return result[columns].drop_duplicates("stock_code", keep="last")


def normalize_listing_events(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "stock_code", "listing_status_event_effective_date", "listing_status_announced_at",
        "listing_status_observable_at", "listing_status_before", "listing_status_after",
        "listing_status_change_type", "listing_status_source_sha256", "listing_status_source_row",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    result["stock_code"] = result["stock_code"].astype(str)
    result["listing_status_event_effective_date"] = pd.to_datetime(result.get("Execudt"), errors="coerce")
    result["listing_status_announced_at"] = pd.to_datetime(result.get("Annoudt"), errors="coerce")
    result["listing_status_observable_at"] = result[[
        "listing_status_event_effective_date", "listing_status_announced_at"
    ]].max(axis=1)
    result["listing_status_before"] = result.get("Stkstatbc", "")
    result["listing_status_after"] = result.get("Stkstatac", "")
    result["listing_status_change_type"] = result.get("Chgtype", "")
    result["listing_status_source_sha256"] = result.get("csmar_source_sha256", "")
    result["listing_status_source_row"] = result.get("source_row_number", pd.NA)
    result = result.dropna(subset=["listing_status_observable_at"])
    return result[columns].sort_values(["stock_code", "listing_status_observable_at"])


def normalize_capital_events(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "stock_code", "capital_effective_date", "capital_change_type",
        "total_shares", "tradable_a_shares", "capital_source_sha256", "capital_source_row",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    result["stock_code"] = result["stock_code"].astype(str)
    result["capital_effective_date"] = pd.to_datetime(result.get("Shrchgdt"), errors="coerce")
    result["capital_change_type"] = result.get("Shrtyp", "")
    result["total_shares"] = pd.to_numeric(result.get("Nshrttl"), errors="coerce")
    result["tradable_a_shares"] = pd.to_numeric(result.get("Nshra"), errors="coerce")
    result["capital_source_sha256"] = result.get("csmar_source_sha256", "")
    result["capital_source_row"] = result.get("source_row_number", pd.NA)
    result = result.dropna(subset=["capital_effective_date"])
    return result[columns].sort_values(["stock_code", "capital_effective_date"])


def load_adjust_factor_events(root: Path) -> pd.DataFrame:
    columns = [
        "stock_code", "adjust_factor_effective_date", "forward_adjust_factor",
        "back_adjust_factor", "adjust_factor", "adjust_factor_source_file",
        "adjust_factor_source_sha256",
    ]
    frames: list[pd.DataFrame] = []
    if root.exists():
        for path in sorted(root.glob("*.adjust_factor.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame = frame.rename(columns={
                "project_stock_code": "stock_code",
                "dividOperateDate": "adjust_factor_effective_date",
                "foreAdjustFactor": "forward_adjust_factor",
                "backAdjustFactor": "back_adjust_factor",
                "adjustFactor": "adjust_factor",
            })
            frame["stock_code"] = frame["stock_code"].astype(str)
            frame["adjust_factor_effective_date"] = pd.to_datetime(
                frame["adjust_factor_effective_date"], errors="coerce"
            )
            for column in ("forward_adjust_factor", "back_adjust_factor", "adjust_factor"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["adjust_factor_source_file"] = path.name
            frame["adjust_factor_source_sha256"] = sha256_file(path)
            frames.append(frame[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).dropna(
        subset=["adjust_factor_effective_date"]
    ).sort_values(["stock_code", "adjust_factor_effective_date"])


def asof_join_by_stock(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    event_date: str,
    rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    if events.empty:
        result = panel.copy()
        for target in (rename or {}).values():
            result[target] = pd.NaT if target.endswith("_date_asof") or target.endswith("_at") else np.nan
        return result
    pieces = []
    event_columns = [column for column in events.columns if column != "stock_code"]
    for stock_code, left in panel.groupby("stock_code", sort=False):
        right = events[events["stock_code"] == stock_code][event_columns].sort_values(event_date)
        left = left.sort_values("trade_date")
        if right.empty:
            joined = left.copy()
            for column in event_columns:
                joined[column] = pd.NaT if column == event_date else np.nan
        else:
            joined = pd.merge_asof(
                left,
                right,
                left_on="trade_date",
                right_on=event_date,
                direction="backward",
                allow_exact_matches=True,
            )
        pieces.append(joined)
    result = pd.concat(pieces, ignore_index=True)
    if rename:
        result = result.rename(columns=rename)
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def dense_panel(base: pd.DataFrame, codes: list[str] | None = None) -> pd.DataFrame:
    selected_codes = sorted(codes or base["stock_code"].unique().tolist())
    dates = pd.Index(sorted(base["trade_date"].unique()))
    index = pd.MultiIndex.from_product([dates, selected_codes], names=KEY_COLUMNS)
    skeleton = index.to_frame(index=False)
    result = skeleton.merge(base, on=KEY_COLUMNS, how="left", validate="one_to_one")
    for column in ("split", "cross_section_coverage", "cross_section_eligible"):
        if column in base:
            by_date = base.dropna(subset=[column]).drop_duplicates("trade_date").set_index("trade_date")[column]
            result[column] = result[column].fillna(result["trade_date"].map(by_date))
    return result


def add_lifecycle(
    panel: pd.DataFrame,
    stock_basic: pd.DataFrame,
    company: pd.DataFrame,
    listing_events: pd.DataFrame,
) -> pd.DataFrame:
    static = stock_basic.merge(company, on="stock_code", how="outer", validate="one_to_one")
    static["listing_date"] = static[["listing_date", "company_listing_date"]].min(axis=1)
    overlapping_static = sorted((set(static.columns) - {"stock_code"}) & set(panel.columns))
    result = panel.drop(columns=overlapping_static).merge(
        static, on="stock_code", how="left", validate="many_to_one"
    )
    result = asof_join_by_stock(
        result,
        listing_events,
        "listing_status_observable_at",
    )
    default_columns = {
        "listing_status_event_effective_date": pd.NaT,
        "listing_status_announced_at": pd.NaT,
        "listing_status_observable_at": pd.NaT,
        "listing_status_before": "",
        "listing_status_after": "",
        "listing_status_change_type": "",
        "listing_status_source_sha256": "",
        "listing_status_source_row": pd.NA,
    }
    for column, default in default_columns.items():
        if column not in result:
            result[column] = default
    result["is_listed_asof"] = result["listing_date"].notna() & result["trade_date"].ge(result["listing_date"])
    result["before_listing_date"] = result["listing_date"].notna() & ~result["is_listed_asof"]
    result["weeks_since_listing"] = (
        (result["trade_date"] - result["listing_date"]).dt.days.div(7).where(result["is_listed_asof"])
    )
    status = result["listing_status_after"].fillna("").astype(str)
    result["listing_status_asof"] = np.where(
        ~result["is_listed_asof"], "not_listed", np.where(status.eq(""), "normal_or_unknown", status)
    )
    result["is_delisted_asof"] = status.str.contains("终止|退市|摘牌|delist", case=False, regex=True)
    result["is_suspended_listing_asof"] = status.str.contains("暂停上市|suspend", case=False, regex=True)
    result["universe_member_pit"] = result["is_listed_asof"] & ~result["is_delisted_asof"]
    return result


def add_capital_and_adjustment(
    panel: pd.DataFrame,
    capital_events: pd.DataFrame,
    adjust_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = asof_join_by_stock(
        panel,
        capital_events,
        "capital_effective_date",
        {
            "capital_effective_date": "capital_effective_date_asof",
            "total_shares": "total_shares_asof",
            "tradable_a_shares": "tradable_a_shares_asof",
        },
    )
    result = asof_join_by_stock(
        result,
        adjust_events,
        "adjust_factor_effective_date",
        {
            "adjust_factor_effective_date": "adjust_factor_effective_date_asof",
            "forward_adjust_factor": "forward_adjust_factor_asof",
            "back_adjust_factor": "back_adjust_factor_asof",
            "adjust_factor": "adjust_factor_asof",
        },
    )
    capital_weeks = set(
        zip(
            capital_events.get("stock_code", pd.Series(dtype=str)),
            pd.to_datetime(capital_events.get("capital_effective_date", pd.Series(dtype="datetime64[ns]")))
            .dt.to_period("W-FRI").dt.end_time.dt.normalize(),
        )
    )
    adjust_weeks = set(
        zip(
            adjust_events.get("stock_code", pd.Series(dtype=str)),
            pd.to_datetime(adjust_events.get("adjust_factor_effective_date", pd.Series(dtype="datetime64[ns]")))
            .dt.to_period("W-FRI").dt.end_time.dt.normalize(),
        )
    )
    keys = list(zip(result["stock_code"], result["trade_date"]))
    result["capital_change_this_week"] = [key in capital_weeks for key in keys]
    result["adjust_factor_change_this_week"] = [key in adjust_weeks for key in keys]
    result["corporate_action_this_week"] = (
        result["capital_change_this_week"] | result["adjust_factor_change_this_week"]
    )
    actions = []
    if not capital_events.empty:
        capital = capital_events.copy()
        capital["action_type"] = "capital_change"
        capital["action_effective_date"] = capital["capital_effective_date"]
        actions.append(capital)
    if not adjust_events.empty:
        adjust = adjust_events.copy()
        adjust["action_type"] = "adjust_factor_change"
        adjust["action_effective_date"] = adjust["adjust_factor_effective_date"]
        actions.append(adjust)
    ledger = pd.concat(actions, ignore_index=True, sort=False) if actions else pd.DataFrame(
        columns=["stock_code", "action_type", "action_effective_date"]
    )
    return result, ledger.sort_values(["stock_code", "action_effective_date"]).reset_index(drop=True)


def add_trading_state(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    observation_column = "model_close" if "model_close" in result else "close"
    volume_column = "model_volume_hands" if "model_volume_hands" in result else "volume_hands"
    result["has_price_observation"] = result[observation_column].notna()
    volume = pd.to_numeric(result.get(volume_column), errors="coerce")
    result["is_zero_volume_observation"] = result["has_price_observation"] & volume.fillna(0).le(0)
    result["is_no_weekly_bar"] = result["universe_member_pit"] & ~result["has_price_observation"]
    result["is_suspended"] = result["universe_member_pit"] & (
        result["is_suspended_listing_asof"] | result["is_zero_volume_observation"]
    )
    result["is_tradable_pit"] = (
        result["universe_member_pit"]
        & result["has_price_observation"]
        & ~result["is_suspended"]
    )
    result["model_eligible_pit"] = result["is_tradable_pit"]
    if "sample_eligible" in result:
        result["sample_eligible_v2"] = result["sample_eligible"].fillna(False).astype(bool) & result["model_eligible_pit"]
    else:
        result["sample_eligible_v2"] = result["model_eligible_pit"]
    special = result.get("csmar_special_status", pd.Series("normal", index=result.index)).fillna("normal").astype(str)
    result["is_special_treatment"] = ~special.str.lower().isin(["normal", "a", "nan", ""])
    result["trade_state"] = np.select(
        [
            ~result["is_listed_asof"],
            result["is_delisted_asof"],
            result["is_suspended"],
            result["is_no_weekly_bar"],
            result["is_tradable_pit"],
        ],
        ["not_listed", "delisted", "suspended", "no_weekly_bar", "tradable"],
        default="inactive_or_unknown",
    )
    return result


def validate_panel_v2(panel: pd.DataFrame) -> dict[str, Any]:
    missing_state = sorted(set(KEY_COLUMNS + STATE_COLUMNS) - set(panel.columns))
    report = {
        "contract_version": CONTRACT_VERSION,
        "rows": int(len(panel)),
        "stocks": int(panel["stock_code"].nunique()),
        "dates": int(panel["trade_date"].nunique()),
        "duplicate_keys": int(panel.duplicated(KEY_COLUMNS).sum()),
        "missing_keys": int(panel[KEY_COLUMNS].isna().any(axis=1).sum()),
        "missing_contract_columns": missing_state,
        "member_before_listing": int((panel["universe_member_pit"] & ~panel["is_listed_asof"]).sum()),
        "member_after_delisting": int((panel["universe_member_pit"] & panel["is_delisted_asof"]).sum()),
        "eligible_without_observation": int((panel["model_eligible_pit"] & ~panel["has_price_observation"]).sum()),
        "eligible_while_suspended": int((panel["model_eligible_pit"] & panel["is_suspended"]).sum()),
        "sample_v2_without_model_eligibility": int((panel["sample_eligible_v2"] & ~panel["model_eligible_pit"]).sum()),
        "universe_member_rows": int(panel["universe_member_pit"].sum()),
        "tradable_rows": int(panel["is_tradable_pit"].sum()),
        "suspended_rows": int(panel["is_suspended"].sum()),
        "no_weekly_bar_rows": int(panel["is_no_weekly_bar"].sum()),
        "corporate_action_rows": int(panel["corporate_action_this_week"].sum()),
    }
    fatal = [
        "duplicate_keys", "missing_keys", "member_before_listing", "member_after_delisting",
        "eligible_without_observation", "eligible_while_suspended",
        "sample_v2_without_model_eligibility",
    ]
    report["fatal_failures"] = [key for key in fatal if report[key] != 0]
    report["passed"] = not report["fatal_failures"] and not missing_state
    return report


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")


def build(config_path: Path, overwrite: bool = False) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    config = read_config(config_path)
    paths = config["paths"]
    output_root = resolve_path(paths["output_root"], project_root)
    if output_root.exists() and any(output_root.iterdir()) and not (overwrite or config.get("overwrite")):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    base_path = resolve_path(paths["base_panel"], project_root)
    base = normalize_base_panel(pd.read_csv(base_path))
    codes = [str(code) for code in config.get("universe", {}).get("codes", [])] or None
    base = base[base["stock_code"].isin(codes)].copy() if codes else base
    panel = dense_panel(base, codes)

    stock_basic_path = resolve_path(paths["stock_basic"], project_root)
    stock_basic = normalize_stock_basic(pd.read_csv(stock_basic_path))
    csmar_root = resolve_path(paths["csmar_prepared_root"], project_root)
    company = normalize_company_records(read_csv_if_exists(csmar_root / "company_records.csv"))
    listing = normalize_listing_events(read_csv_if_exists(csmar_root / "listing_status_events.csv"))
    capital = normalize_capital_events(read_csv_if_exists(csmar_root / "capital_events_all_history.csv"))
    adjust_root = resolve_path(paths["baostock_adjust_root"], project_root)
    adjust = load_adjust_factor_events(adjust_root)

    panel = add_lifecycle(panel, stock_basic, company, listing)
    panel, corporate_actions = add_capital_and_adjustment(panel, capital, adjust)
    panel = add_trading_state(panel)
    panel = panel.sort_values(KEY_COLUMNS).reset_index(drop=True)
    action_count_all_history = len(corporate_actions)
    corporate_actions = corporate_actions[
        pd.to_datetime(corporate_actions["action_effective_date"], errors="coerce")
        .le(panel["trade_date"].max())
    ].copy()
    report = validate_panel_v2(panel)
    if not report["passed"]:
        raise ValueError(f"panel_v2 validation failed: {report}")

    membership_columns = KEY_COLUMNS + [column for column in STATE_COLUMNS if column in panel.columns]
    membership = panel[membership_columns].copy()
    write_csv_gz(panel, output_root / "panel_v2.csv.gz")
    write_csv_gz(membership, output_root / "universe_membership.csv.gz")
    write_csv_gz(corporate_actions, output_root / "corporate_actions.csv.gz")
    (output_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contract_path = resolve_path(config["contract_schema"], project_root)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    (output_root / "contract_snapshot.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": stable_json_hash(config),
        "base_panel": {"path": str(base_path.resolve()), "sha256": sha256_file(base_path)},
        "stock_basic": {"path": str(stock_basic_path.resolve()), "sha256": sha256_file(stock_basic_path)},
        "csmar_prepared_root": str(csmar_root.resolve()),
        "baostock_adjust_root": str(adjust_root.resolve()),
        "grain": "one row per canonical trade week and stock code",
        "key": KEY_COLUMNS,
        "canonical_trade_date": "calendar week ending Friday; observation_trade_date preserves actual source date",
        "point_in_time_policy": {
            "listing_status": "event becomes active only when both announced and effective",
            "capital": "effective date is used as availability because export has no announcement timestamp",
            "suspension": "zero-volume observation is strict suspension evidence; a missing weekly bar remains a separate no_weekly_bar state",
            "universe": "suspended stocks remain universe members but are not tradable/model eligible",
            "snapshot_fields": "current stock_basic listing status is audit-only and never backfilled into historical membership",
        },
        "selection_bias_warning": config.get("universe", {}).get("selection_bias_warning"),
        "corporate_action_ledger": {
            "all_history_event_count_seen_during_asof_build": action_count_all_history,
            "development_ledger_event_count_written": len(corporate_actions),
            "written_through": panel["trade_date"].max().date().isoformat(),
            "future_events_written": 0,
        },
        "validation": report,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("data_pipeline/configs/panel_v2_30stocks.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    print(build(config_path, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
