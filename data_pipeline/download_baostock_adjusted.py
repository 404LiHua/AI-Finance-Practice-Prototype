"""Download traceable BaoStock forward-adjusted weekly data and adjust factors."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import baostock as bs
import pandas as pd


DEFAULT_CODES = ["000001.SZ", "000002.SZ", "000006.SZ", "000007.SZ", "000008.SZ"]
FIELDS = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"


def to_baostock_code(code: str) -> str:
    number, exchange = code.split(".")
    return f"{exchange.lower()}.{number}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def result_to_frame(result) -> pd.DataFrame:
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock query failed: {result.error_code} {result.error_msg}")
    return pd.DataFrame(rows, columns=result.fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data_pipeline/configs/weekly_a_share.json"))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--codes", nargs="*")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = {}
    if args.config:
        config_path = args.config if args.config.is_absolute() else project_root / args.config
        config = json.loads(config_path.read_text(encoding="utf-8"))
    args.start = args.start or config.get("date_range", {}).get("start") or "2022-06-03"
    args.end = args.end or config.get("date_range", {}).get("end") or "2023-06-02"
    args.codes = args.codes or config.get("universe", {}).get("codes") or DEFAULT_CODES
    configured_output = config.get("baostock", {}).get("root")
    args.output = args.output or Path(configured_output or "data/external/baostock_download")
    if not args.output.is_absolute():
        args.output = project_root / args.output
    args.output.mkdir(parents=True, exist_ok=True)

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
    files = []
    try:
        for code in args.codes:
            api_code = to_baostock_code(code)
            weekly = result_to_frame(bs.query_history_k_data_plus(
                api_code,
                FIELDS,
                start_date=args.start,
                end_date=args.end,
                frequency="w",
                adjustflag="2",
            ))
            weekly.insert(0, "project_stock_code", code)
            weekly_path = args.output / f"{code}.weekly_qfq.csv"
            weekly.to_csv(weekly_path, index=False, encoding="utf-8-sig")
            files.append({
                "file": weekly_path.name,
                "type": "weekly_forward_adjusted",
                "rows": len(weekly),
                "sha256": sha256_file(weekly_path),
            })

            factors = result_to_frame(bs.query_adjust_factor(api_code, start_date=args.start, end_date=args.end))
            factors.insert(0, "project_stock_code", code)
            factor_path = args.output / f"{code}.adjust_factor.csv"
            factors.to_csv(factor_path, index=False, encoding="utf-8-sig")
            files.append({
                "file": factor_path.name,
                "type": "adjust_factor",
                "rows": len(factors),
                "sha256": sha256_file(factor_path),
            })
    finally:
        bs.logout()

    manifest = {
        "provider": "BaoStock",
        "package_version": getattr(bs, "__version__", "0.9.3"),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "codes": args.codes,
        "start_date": args.start,
        "end_date": args.end,
        "weekly_request": {
            "frequency": "w",
            "adjustflag": "2",
            "adjustment": "前复权",
            "fields": FIELDS.split(","),
        },
        "files": files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
