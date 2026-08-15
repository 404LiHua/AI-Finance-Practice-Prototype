from __future__ import annotations

"""Audit normalized historical corporate actions against the raw CSMAR archive."""

import argparse
import hashlib
import importlib.util
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


def load_parser(path: Path):
    spec = importlib.util.spec_from_file_location("wp22_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import parser from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_xlsx_table


def number(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--csmar-zip", required=True, type=Path)
    parser.add_argument("--selected-universe", required=True, type=Path)
    parser.add_argument("--parser-script", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    historical = args.historical.resolve(); raw_zip = args.csmar_zip.resolve(); universe_path = args.selected_universe.resolve(); output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output {output}")
    if not all(path.is_file() for path in (historical, raw_zip, universe_path, args.parser_script.resolve())):
        raise RuntimeError("required audit input missing")

    selected = set(pd.read_csv(universe_path, dtype={"stock_code": str})["stock_code"].astype(str))
    archived = pd.read_csv(historical, dtype={"stock_code": str})
    archived["stock_code"] = archived["stock_code"].astype(str)
    archived["capital_effective_date"] = pd.to_datetime(archived["capital_effective_date"], errors="raise").dt.normalize()
    archived["Shrtyp"] = archived["Shrtyp"].astype(str).str.strip().str.zfill(5)
    archived = archived[archived["stock_code"].isin(selected)].copy()

    parse_xlsx_table = load_parser(args.parser_script.resolve())
    rows = parse_xlsx_table(raw_zip, "SPT_Capchg")
    normalized = []
    for row_number, row in enumerate(rows):
        code6 = str(row.get("Stkcd", "")).strip().zfill(6)
        code = next((item for item in selected if item.startswith(code6 + ".")), None)
        date = pd.to_datetime(row.get("Shrchgdt", ""), errors="coerce")
        if code is None or pd.isna(date):
            continue
        normalized.append({
            "stock_code": code,
            "capital_effective_date": pd.Timestamp(date).normalize(),
            "Shrtyp": str(row.get("Shrtyp", "")).strip().zfill(5),
            "total_shares": number(row.get("Nshrttl")),
            "tradable_a_shares": number(row.get("Nshra")),
            "source_row_number_raw_parser": row_number + 2,
        })
    raw = pd.DataFrame(normalized)
    if raw.empty:
        raise RuntimeError("no selected-universe CSMAR capital rows parsed")

    keys = ["stock_code", "capital_effective_date", "Shrtyp", "total_shares", "tradable_a_shares"]
    left = archived[keys].copy(); right = raw[keys].copy()
    left["total_shares"] = pd.to_numeric(left["total_shares"], errors="coerce"); left["tradable_a_shares"] = pd.to_numeric(left["tradable_a_shares"], errors="coerce")
    right["total_shares"] = pd.to_numeric(right["total_shares"], errors="coerce"); right["tradable_a_shares"] = pd.to_numeric(right["tradable_a_shares"], errors="coerce")
    left["_archived"] = True; right["_raw"] = True
    merged = left.merge(right, on=keys, how="outer", indicator=True)
    only_archived = merged[merged["_merge"].eq("left_only")].copy()
    only_raw = merged[merged["_merge"].eq("right_only")].copy()
    # Also test the archive's provenance row numbers against the parser's row numbers.
    provenance = archived[["stock_code", "capital_effective_date", "Shrtyp", "source_row_number"]].merge(raw, on=["stock_code", "capital_effective_date", "Shrtyp"], how="left")
    provenance["row_number_match"] = provenance["source_row_number"].eq(provenance["source_row_number_raw_parser"])
    output.mkdir(parents=True)
    only_archived.to_csv(output / "ARCHIVED_ONLY_ROWS.csv", index=False, encoding="utf-8-sig")
    only_raw.to_csv(output / "RAW_ONLY_ROWS.csv", index=False, encoding="utf-8-sig")
    provenance.to_csv(output / "PROVENANCE_ROW_NUMBER_AUDIT.csv", index=False, encoding="utf-8-sig")
    historical_end = archived["capital_effective_date"].max()
    raw_only_after_historical_end = bool(len(only_raw) == 0 or (pd.to_datetime(only_raw["capital_effective_date"], errors="raise") > historical_end).all())
    historical_exact = len(only_archived) == 0
    decision = {
        "node_id": "WP22_CSMAR_CAPITAL_ACTIONS_OVERLAP_AUDIT_V1",
        "status": "PASS_HISTORICAL_EXACT_RAW_EXTENDED" if historical_exact and raw_only_after_historical_end else "FAIL_OVERLAP_MISMATCH",
        "historical_path": str(historical), "csmar_zip": str(raw_zip),
        "historical_sha256": sha256(historical), "csmar_zip_sha256": sha256(raw_zip), "selected_universe_sha256": sha256(universe_path),
        "historical_selected_rows": int(len(left)), "raw_selected_rows": int(len(right)),
        "exact_overlap_rows": int(len(merged[merged["_merge"].eq("both")])),
        "archived_only_rows": int(len(only_archived)), "raw_only_rows": int(len(only_raw)),
        "historical_end_date": historical_end.date().isoformat(), "raw_only_after_historical_end": raw_only_after_historical_end,
        "provenance_rows": int(len(provenance)), "provenance_row_number_matches": int(provenance["row_number_match"].sum()),
        "provenance_row_number_mismatches": int((~provenance["row_number_match"]).sum()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "CSMAR_CAPITAL_ACTIONS_OVERLAP_AUDIT.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
