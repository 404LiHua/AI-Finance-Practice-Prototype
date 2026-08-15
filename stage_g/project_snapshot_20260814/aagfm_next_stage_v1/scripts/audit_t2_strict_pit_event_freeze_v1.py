from __future__ import annotations

"""CPU-only, label-free preconsumption audit for a production-T2 event source freeze."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_columns(frame: pd.DataFrame, name: str, columns: set[str], failures: list[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        failures.append(f"{name}_missing_columns:{','.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--origins", required=True, type=Path)
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--sha256-manifest", required=True, type=Path)
    parser.add_argument("--expected-origin-registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    paths = {key: getattr(args, key).resolve() for key in ("freeze_manifest", "origins", "membership", "coverage", "events", "sha256_manifest")}
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")

    manifest = json.loads(paths["freeze_manifest"].read_text(encoding="utf-8"))
    origins = pd.read_csv(paths["origins"], dtype=str)
    membership = pd.read_csv(paths["membership"], dtype=str)
    coverage = pd.read_csv(paths["coverage"], dtype=str)
    events = pd.read_parquet(paths["events"])
    sha_manifest = pd.read_csv(paths["sha256_manifest"], dtype=str)
    expected_origin_registry_path = args.expected_origin_registry.resolve()
    expected_origin_registry = pd.read_csv(expected_origin_registry_path, dtype=str)
    failures: list[str] = []
    if manifest.get("status") != "FROZEN_BEFORE_T2_EVENT_PRECONSUMPTION":
        failures.append("freeze_manifest_status")
    if manifest.get("target_id") != "T2_MARKET_RELATIVE_FIXED":
        failures.append("target_id")
    if manifest.get("timezone") != "Asia/Shanghai" or not manifest.get("cutoff_authority"):
        failures.append("cutoff_authority")
    if manifest.get("strict_inclusion_rule") not in {"published_at_utc <= cutoff_at_utc", "published_at_utc < cutoff_at_utc"}:
        failures.append("strict_inclusion_rule")
    required_columns(origins, "origins", {"trade_date", "cutoff_at_utc", "cutoff_rule_id"}, failures)
    required_columns(membership, "membership", {"trade_date", "stock_code", "eligible", "membership_effective_at"}, failures)
    required_columns(coverage, "coverage", {"stock_code", "coverage_start_date", "coverage_end_date", "coverage_status", "source_system", "source_snapshot_sha256"}, failures)
    required_columns(events, "events", {"event_id", "stock_code", "published_at_utc", "source_response_sha256", "source_url"}, failures)
    required_columns(sha_manifest, "sha256_manifest", {"relative_path", "sha256"}, failures)
    required_columns(expected_origin_registry, "expected_origin_registry", {"trade_date"}, failures)

    origin_dates = pd.Series(dtype="datetime64[ns]")
    cutoffs = pd.Series(dtype="datetime64[ns, UTC]")
    event_times = pd.Series(dtype="datetime64[ns, UTC]")
    try:
        if {"trade_date", "cutoff_at_utc"}.issubset(origins.columns):
            origin_dates = pd.to_datetime(origins["trade_date"], errors="raise", utc=False)
            cutoffs = pd.to_datetime(origins["cutoff_at_utc"], errors="raise", utc=True)
        if "published_at_utc" in events.columns:
            event_times = pd.to_datetime(events["published_at_utc"], errors="raise", utc=True)
    except (KeyError, ValueError, TypeError) as error:
        failures.append(f"timestamp_parse:{type(error).__name__}")
    if not origins.empty and {"trade_date", "cutoff_at_utc", "cutoff_rule_id"}.issubset(origins.columns):
        if origins["trade_date"].duplicated().any() or origins["cutoff_at_utc"].isna().any() or origins["cutoff_rule_id"].isna().any():
            failures.append("origin_key_or_cutoff")
        if len(origin_dates) and not bool((origin_dates.dt.weekday == 4).all()):
            failures.append("non_friday_t2_origin")
        if len(cutoffs) and bool((cutoffs.dt.tz_convert("Asia/Shanghai").dt.date != origin_dates.dt.date).any()):
            failures.append("cutoff_not_on_origin_trade_date")
    if {"trade_date"}.issubset(expected_origin_registry.columns):
        expected_dates = pd.to_datetime(expected_origin_registry["trade_date"], errors="coerce")
        if expected_dates.isna().any() or expected_origin_registry["trade_date"].duplicated().any():
            failures.append("expected_origin_registry_key")
        elif set(expected_origin_registry["trade_date"].astype(str)) != set(origins.get("trade_date", pd.Series(dtype=str)).astype(str)):
            failures.append("origin_registry_mismatch")
    membership_effective = pd.Series(dtype="datetime64[ns, UTC]")
    if {"trade_date", "stock_code", "eligible", "membership_effective_at"}.issubset(membership.columns):
        membership_dates = pd.to_datetime(membership["trade_date"], errors="coerce")
        membership_effective = pd.to_datetime(membership["membership_effective_at"], errors="coerce", utc=True)
        if (
            membership.duplicated(["trade_date", "stock_code"]).any()
            or membership["eligible"].isna().any()
            or membership["stock_code"].astype(str).str.strip().eq("").any()
            or membership_dates.isna().any()
            or membership_effective.isna().any()
        ):
            failures.append("membership_key_or_eligibility")
        if {"trade_date"}.issubset(origins.columns):
            origin_date_keys = set(origins["trade_date"].astype(str))
            membership_date_keys = set(membership["trade_date"].astype(str))
            if origin_date_keys.difference(membership_date_keys):
                failures.append("origin_without_membership_rows")
        if len(cutoffs) and not membership_dates.isna().any() and not membership_effective.isna().any():
            cutoff_by_date = dict(zip(origins["trade_date"].astype(str), cutoffs))
            member_cutoffs = membership["trade_date"].astype(str).map(cutoff_by_date)
            if member_cutoffs.isna().any() or bool((membership_effective > member_cutoffs).any()):
                failures.append("membership_not_effective_by_cutoff")
    if {"event_id", "published_at_utc", "stock_code", "source_url"}.issubset(events.columns):
        if (
            events["event_id"].duplicated().any()
            or events["published_at_utc"].isna().any()
            or events["stock_code"].isna().any()
            or events["stock_code"].astype(str).str.strip().eq("").any()
            or events["source_url"].astype(str).str.strip().eq("").any()
        ):
            failures.append("event_key_timestamp_or_source_url")
    response_hashes = events.get("source_response_sha256", pd.Series(dtype=str)).astype(str)
    coverage_hashes = coverage.get("source_snapshot_sha256", pd.Series(dtype=str)).astype(str)
    if not response_hashes.map(lambda value: bool(SHA256_RE.fullmatch(value.lower()))).all() or not coverage_hashes.map(lambda value: bool(SHA256_RE.fullmatch(value.lower()))).all():
        failures.append("source_sha256_format")
    if {"coverage_start_date", "coverage_end_date", "coverage_status"}.issubset(coverage.columns):
        coverage_starts = pd.to_datetime(coverage["coverage_start_date"], errors="coerce")
        coverage_ends = pd.to_datetime(coverage["coverage_end_date"], errors="coerce")
        valid_coverage_statuses = {"COVERED", "NO_EVENTS"}
        if (
            coverage_starts.isna().any()
            or coverage_ends.isna().any()
            or bool((coverage_starts > coverage_ends).any())
            or not coverage["coverage_status"].astype(str).str.upper().isin(valid_coverage_statuses).all()
        ):
            failures.append("coverage_window_or_status")
    if len(event_times) and len(cutoffs) and bool((event_times > cutoffs.max()).any()):
        failures.append("event_after_maximum_origin_cutoff")

    eligible = pd.DataFrame(columns=["trade_date", "stock_code"])
    covered_eligible = pd.DataFrame(columns=["trade_date", "stock_code"])
    if {"trade_date", "stock_code", "eligible"}.issubset(membership.columns) and {"stock_code", "coverage_start_date", "coverage_end_date", "coverage_status"}.issubset(coverage.columns):
        eligible = membership.loc[membership["eligible"].astype(str).str.lower().isin({"true", "1", "yes"}), ["trade_date", "stock_code"]].copy()
        eligible["trade_date"] = pd.to_datetime(eligible["trade_date"], errors="coerce")
        coverage_window = coverage.loc[coverage["coverage_status"].astype(str).str.upper().isin({"COVERED", "NO_EVENTS"}), ["stock_code", "coverage_start_date", "coverage_end_date"]].copy()
        coverage_window["coverage_start_date"] = pd.to_datetime(coverage_window["coverage_start_date"], errors="coerce")
        coverage_window["coverage_end_date"] = pd.to_datetime(coverage_window["coverage_end_date"], errors="coerce")
        if eligible["trade_date"].isna().any() or coverage_window[["coverage_start_date", "coverage_end_date"]].isna().any().any():
            failures.append("membership_or_coverage_date_parse")
        else:
            eligible["membership_row_id"] = range(len(eligible))
            candidates = eligible.merge(coverage_window, on="stock_code", how="left")
            candidates = candidates[(candidates["coverage_start_date"] <= candidates["trade_date"]) & (candidates["trade_date"] <= candidates["coverage_end_date"])]
            covered_eligible = candidates[["membership_row_id", "trade_date", "stock_code"]].drop_duplicates("membership_row_id")
    eligible_codes = set(eligible["stock_code"].astype(str))
    covered_codes = set(covered_eligible["stock_code"].astype(str))
    uncovered = sorted(eligible_codes.difference(covered_codes))
    if len(covered_eligible) != len(eligible):
        failures.append("eligible_universe_not_fully_covered")
    audited_paths = {key: path for key, path in paths.items() if key != "sha256_manifest"}
    computed_hashes = {path.name: sha256(path) for path in audited_paths.values()}
    sha256_manifest_sha256 = sha256(paths["sha256_manifest"])
    declared = dict(zip(sha_manifest.get("relative_path", []), sha_manifest.get("sha256", [])))
    if sha_manifest.duplicated("relative_path").any() or not sha_manifest.get("sha256", pd.Series(dtype=str)).astype(str).map(lambda value: bool(SHA256_RE.fullmatch(value.lower()))).all():
        failures.append("sha256_manifest_key_or_format")
    manifest_mismatches = [name for name, digest in computed_hashes.items() if declared.get(name, "").lower() != digest]
    if manifest_mismatches:
        failures.append("sha256_manifest_mismatch")

    result = {
        "node_id": "T2_STRICT_PIT_EVENT_FREEZE_PRECONSUMPTION_AUDIT_V1",
        "status": "PASS_T2_EVENT_FREEZE_READY_FOR_SEPARATE_TRAIN_ONLY_AUTHORIZATION" if not failures else "FAIL_CLOSED_T2_EVENT_FREEZE_PRECONSUMPTION",
        "input_sha256": computed_hashes,
        "expected_origin_registry_sha256": sha256(expected_origin_registry_path),
        "sha256_manifest_sha256": sha256_manifest_sha256,
        "origin_count": int(len(origins)),
        "eligible_stock_count": int(len(eligible_codes)),
        "covered_stock_count": int(len(eligible_codes.intersection(covered_codes))),
        "eligible_membership_row_count": int(len(eligible)),
        "covered_eligible_membership_row_count": int(len(covered_eligible)),
        "uncovered_stock_count": int(len(uncovered)),
        "uncovered_stock_sample": uncovered[:20],
        "event_count": int(len(events)),
        "events_after_latest_cutoff": int((event_times > cutoffs.max()).sum()) if len(event_times) and len(cutoffs) else 0,
        "labels_payload_read": False,
        "fresh_payload_read": False,
        "screening_read": False,
        "final_read": False,
        "model_trained": False,
        "gpu_used": False,
        "failures": failures,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True)
    (output / "T2_EVENT_FREEZE_PRECONSUMPTION_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


