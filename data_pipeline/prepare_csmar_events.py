"""Normalize CSMAR special-treatment ZIP exports into weekly model inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_CODES = ["000001.SZ", "000002.SZ", "000006.SZ", "000007.SZ", "000008.SZ"]
STATUS_NAMES = {
    "A": "normal",
    "B": "ST",
    "D": "*ST",
    "C": "PT",
    "S": "suspended_listing",
    "T": "delisting_period",
    "X": "delisted",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_code(value) -> str:
    text = re.sub(r"\.0$", "", str(value).strip())
    return text.zfill(6) if text.isdigit() else text


def project_code(six_digit_code: str) -> str:
    exchange = "SH" if six_digit_code.startswith(("5", "6", "9")) else "SZ"
    return f"{six_digit_code}.{exchange}"


def read_zip_table(root: Path, zip_pattern: str, member: str) -> tuple[pd.DataFrame, Path]:
    matches = list(root.glob(zip_pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {zip_pattern}, found {len(matches)}")
    zip_path = matches[0]
    with zipfile.ZipFile(zip_path) as archive:
        frame = pd.read_excel(io.BytesIO(archive.read(member)), dtype=str)
    frame["source_row_number"] = range(2, len(frame) + 2)
    frame["normalized_code"] = frame.iloc[:, 0].map(normalize_code)
    frame = frame[frame["normalized_code"].str.fullmatch(r"\d{6}", na=False)].copy()
    frame["stock_code"] = frame["normalized_code"].map(project_code)
    frame["csmar_source_zip"] = zip_path.name
    frame["csmar_source_sha256"] = sha256_file(zip_path)
    return frame, zip_path


def prepare(args) -> Path:
    args.output.mkdir(parents=True, exist_ok=True)
    selected = set(args.codes)
    special, special_zip = read_zip_table(args.source, "*\u7279\u6b8a\u5904\u7406*.zip", "SPT_Trdchg.xlsx")
    listing, listing_zip = read_zip_table(args.source, "*\u4e0a\u5e02\u72b6\u6001*.zip", "SPT_LTDSTACHG.xlsx")
    capital, capital_zip = read_zip_table(args.source, "*\u80a1\u672c\u7ed3\u6784*.zip", "SPT_Capchg.xlsx")
    company, company_zip = read_zip_table(args.source, "*\u516c\u53f8\u6587\u4ef6*.zip", "SPT_Company.xlsx")
    special = special[special["stock_code"].isin(selected)].copy()
    listing = listing[listing["stock_code"].isin(selected)].copy()
    capital = capital[capital["stock_code"].isin(selected)].copy()
    company = company[company["stock_code"].isin(selected)].copy()
    for frame, columns in (
        (special, ["Annoudt", "Execudt"]),
        (listing, ["Annoudt", "Execudt"]),
        (capital, ["Shrchgdt"]),
        (company, ["Listdt", "Statdt"]),
    ):
        for column in columns:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    week_ends = pd.date_range(start, end, freq="W-FRI")
    weekly_rows = []
    for stock_code in args.codes:
        stock_special = special[special["stock_code"] == stock_code].sort_values("Execudt")
        stock_capital = capital[capital["stock_code"] == stock_code].sort_values("Shrchgdt")
        for week_end in week_ends:
            previous_special = stock_special[stock_special["Execudt"] <= week_end]
            previous_capital = stock_capital[stock_capital["Shrchgdt"] <= week_end]
            status = "normal"
            status_date = pd.NaT
            status_source = ""
            status_hash = ""
            status_row = pd.NA
            if not previous_special.empty:
                latest = previous_special.iloc[-1]
                status_code = str(latest.get("Chgtype", "A"))[-1:]
                status = STATUS_NAMES.get(status_code, status_code or "unknown")
                status_date = latest["Execudt"]
                status_source = latest["csmar_source_zip"]
                status_hash = latest["csmar_source_sha256"]
                status_row = latest["source_row_number"]
            total_shares = pd.NA
            tradable_a_shares = pd.NA
            capital_date = pd.NaT
            if not previous_capital.empty:
                latest_cap = previous_capital.iloc[-1]
                total_shares = pd.to_numeric(latest_cap.get("Nshrttl"), errors="coerce")
                tradable_a_shares = pd.to_numeric(latest_cap.get("Nshra"), errors="coerce")
                capital_date = latest_cap["Shrchgdt"]
            weekly_rows.append({
                "stock_code": stock_code,
                "calendar_week_end": week_end,
                "csmar_special_status": status,
                "csmar_special_status_date": status_date,
                "csmar_total_shares": total_shares,
                "csmar_tradable_a_shares": tradable_a_shares,
                "csmar_capital_effective_date": capital_date,
                "csmar_status_source_zip": status_source,
                "csmar_status_source_sha256": status_hash,
                "csmar_status_source_row": status_row,
            })
    weekly = pd.DataFrame(weekly_rows)

    event_rows = []
    special_window = special[special["Annoudt"].between(start, end)].copy()
    for _, row in special_window.iterrows():
        reason = "" if pd.isna(row.get("Chgrsdis")) else str(row.get("Chgrsdis"))
        content = "" if pd.isna(row.get("Content")) else str(row.get("Content"))
        before = "" if pd.isna(row.get("Stknmebc")) else str(row.get("Stknmebc"))
        after = "" if pd.isna(row.get("Stknmeac")) else str(row.get("Stknmeac"))
        event_rows.append({
            "published_at": row["Annoudt"],
            "stock_code": row["stock_code"],
            "title": f"Special treatment change: {before} -> {after}",
            "body": f"effective_date={row['Execudt']}; change_type={row.get('Chgtype','')}; reason={reason}; content={content}",
            "source": "CSMAR SPT_Trdchg",
            "url": f"{row['csmar_source_zip']}#row={row['source_row_number']}",
            "event_type": "special_treatment",
            "source_sha256": row["csmar_source_sha256"],
            "source_row_number": row["source_row_number"],
        })
    capital_window = capital[capital["Shrchgdt"].between(start, end)].copy()
    for _, row in capital_window.iterrows():
        event_rows.append({
            "published_at": row["Shrchgdt"],
            "stock_code": row["stock_code"],
            "title": "Capital structure change",
            "body": f"change_type={row.get('Shrtyp','')}; total_shares={row.get('Nshrttl','')}; tradable_a_shares={row.get('Nshra','')}",
            "source": "CSMAR SPT_Capchg",
            "url": f"{row['csmar_source_zip']}#row={row['source_row_number']}",
            "event_type": "capital_change",
            "source_sha256": row["csmar_source_sha256"],
            "source_row_number": row["source_row_number"],
        })
    events = pd.DataFrame(event_rows, columns=[
        "published_at", "stock_code", "title", "body", "source", "url",
        "event_type", "source_sha256", "source_row_number",
    ]).sort_values(["published_at", "stock_code"])

    weekly.to_csv(args.output / "weekly_features.csv", index=False, encoding="utf-8-sig")
    events.to_csv(args.output / "text_events.csv", index=False, encoding="utf-8-sig")
    special.to_csv(args.output / "special_events_all_history.csv", index=False, encoding="utf-8-sig")
    capital.to_csv(args.output / "capital_events_all_history.csv", index=False, encoding="utf-8-sig")
    company.to_csv(args.output / "company_records.csv", index=False, encoding="utf-8-sig")
    listing.to_csv(args.output / "listing_status_events.csv", index=False, encoding="utf-8-sig")
    output_files = []
    for path in sorted(args.output.glob("*.csv")):
        output_files.append({"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "provider": "CSMAR",
        "usage_notice": "Export marked for Jiaxing University use only",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source.resolve()),
        "codes": args.codes,
        "start_date": args.start,
        "end_date": args.end,
        "source_zips": [
            {"file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (special_zip, listing_zip, capital_zip, company_zip)
        ],
        "event_counts": events["event_type"].value_counts().to_dict(),
        "weekly_status_counts": weekly["csmar_special_status"].value_counts().to_dict(),
        "outputs": output_files,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data_pipeline/configs/weekly_a_share.json"))
    parser.add_argument("--source", type=Path, default=Path("D:/\u9879\u76ee/data/csmar"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--codes", nargs="*")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args.start = args.start or config["date_range"]["start"]
    args.end = args.end or config["date_range"]["end"]
    args.codes = args.codes or config["universe"]["codes"] or DEFAULT_CODES
    if args.output is None:
        manifest_path = Path(config["csmar"]["manifest_path"])
        args.output = manifest_path.parent
    if not args.output.is_absolute():
        args.output = project_root / args.output
    prepare(args)


if __name__ == "__main__":
    main()
