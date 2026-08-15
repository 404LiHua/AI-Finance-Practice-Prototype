from __future__ import annotations

"""Synthetic, label-free acceptance tests for the T2 event-freeze preconsumption auditor."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sha_manifest(root: Path, files: list[Path]) -> None:
    pd.DataFrame({"relative_path": [path.name for path in files], "sha256": [sha256(path) for path in files]}).to_csv(
        root / "SHA256_MANIFEST.csv", index=False
    )


def invoke(root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).with_name("audit_t2_strict_pit_event_freeze_v1.py")
    return subprocess.run(
        [
            sys.executable, str(script),
            "--freeze-manifest", str(root / "FREEZE_MANIFEST.json"),
            "--origins", str(root / "T2_ORIGIN_CUTOFFS.csv"),
            "--membership", str(root / "PIT_UNIVERSE_MEMBERSHIP.csv"),
            "--coverage", str(root / "SOURCE_COVERAGE_RECEIPTS.csv"),
            "--events", str(root / "STRICT_PIT_EVENTS.parquet"),
            "--sha256-manifest", str(root / "SHA256_MANIFEST.csv"),
            "--expected-origin-registry", str(root / "T2_PRODUCTION_TRAIN_ORIGIN_REGISTRY_V1.csv"),
            "--output-root", str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="t2_event_freeze_test_") as temporary:
        root = Path(temporary)
        (root / "FREEZE_MANIFEST.json").write_text(json.dumps({
            "status": "FROZEN_BEFORE_T2_EVENT_PRECONSUMPTION",
            "freeze_id": "TEST_ONLY",
            "target_id": "T2_MARKET_RELATIVE_FIXED",
            "timezone": "Asia/Shanghai",
            "cutoff_authority": "test authority",
            "strict_inclusion_rule": "published_at_utc <= cutoff_at_utc",
        }), encoding="utf-8")
        pd.DataFrame({
            "trade_date": ["2020-01-03", "2020-01-10"],
            "cutoff_at_utc": ["2020-01-03T01:30:00Z", "2020-01-10T01:30:00Z"],
            "cutoff_rule_id": ["FRI_0930", "FRI_0930"],
        }).to_csv(root / "T2_ORIGIN_CUTOFFS.csv", index=False)
        pd.DataFrame({"trade_date": ["2020-01-03", "2020-01-10"]}).to_csv(
            root / "T2_PRODUCTION_TRAIN_ORIGIN_REGISTRY_V1.csv", index=False
        )
        pd.DataFrame({
            "trade_date": ["2020-01-03", "2020-01-03", "2020-01-10", "2020-01-10"],
            "stock_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "eligible": [True, True, True, True],
            "membership_effective_at": ["2019-01-01T00:00:00Z"] * 4,
        }).to_csv(root / "PIT_UNIVERSE_MEMBERSHIP.csv", index=False)
        pd.DataFrame({
            "stock_code": ["000001.SZ", "000002.SZ"],
            "coverage_start_date": ["2019-01-01", "2019-01-01"],
            "coverage_end_date": ["2020-12-31", "2020-12-31"],
            "coverage_status": ["COVERED", "NO_EVENTS"],
            "source_system": ["test", "test"],
            "source_snapshot_sha256": ["a" * 64, "b" * 64],
        }).to_csv(root / "SOURCE_COVERAGE_RECEIPTS.csv", index=False)
        pd.DataFrame({
            "event_id": ["event-1"], "stock_code": ["000001.SZ"],
            "published_at_utc": ["2020-01-02T01:00:00Z"],
            "source_response_sha256": ["c" * 64], "source_url": ["https://example.invalid/event-1"],
        }).to_parquet(root / "STRICT_PIT_EVENTS.parquet", index=False)
        inputs = [
            root / "FREEZE_MANIFEST.json", root / "T2_ORIGIN_CUTOFFS.csv", root / "PIT_UNIVERSE_MEMBERSHIP.csv",
            root / "SOURCE_COVERAGE_RECEIPTS.csv", root / "STRICT_PIT_EVENTS.parquet",
        ]
        write_sha_manifest(root, inputs)
        passed = invoke(root, root / "out_pass")
        if passed.returncode != 0:
            raise RuntimeError(f"expected pass, got {passed.returncode}: {passed.stdout} {passed.stderr}")
        receipt = json.loads((root / "out_pass" / "T2_EVENT_FREEZE_PRECONSUMPTION_AUDIT.json").read_text(encoding="utf-8"))
        if receipt["status"] != "PASS_T2_EVENT_FREEZE_READY_FOR_SEPARATE_TRAIN_ONLY_AUTHORIZATION" or receipt["labels_payload_read"]:
            raise RuntimeError("unexpected passing receipt")

        pd.DataFrame({"trade_date": ["2020-01-03"]}).to_csv(root / "T2_PRODUCTION_TRAIN_ORIGIN_REGISTRY_V1.csv", index=False)
        registry_mismatch = invoke(root, root / "out_registry_mismatch")
        receipt = json.loads((root / "out_registry_mismatch" / "T2_EVENT_FREEZE_PRECONSUMPTION_AUDIT.json").read_text(encoding="utf-8"))
        if registry_mismatch.returncode != 2 or "origin_registry_mismatch" not in receipt["failures"]:
            raise RuntimeError(f"expected origin-registry failure: {registry_mismatch.returncode} {receipt}")
        pd.DataFrame({"trade_date": ["2020-01-03", "2020-01-10"]}).to_csv(
            root / "T2_PRODUCTION_TRAIN_ORIGIN_REGISTRY_V1.csv", index=False
        )

        pd.DataFrame({
            "trade_date": ["2020-01-03", "2020-01-03"],
            "stock_code": ["000001.SZ", "000002.SZ"],
            "eligible": [True, True],
            "membership_effective_at": ["2019-01-01T00:00:00Z"] * 2,
        }).to_csv(root / "PIT_UNIVERSE_MEMBERSHIP.csv", index=False)
        write_sha_manifest(root, inputs)
        missing_origin = invoke(root, root / "out_missing_origin")
        receipt = json.loads((root / "out_missing_origin" / "T2_EVENT_FREEZE_PRECONSUMPTION_AUDIT.json").read_text(encoding="utf-8"))
        if missing_origin.returncode != 2 or "origin_without_membership_rows" not in receipt["failures"]:
            raise RuntimeError(f"expected missing-origin failure: {missing_origin.returncode} {receipt}")

        pd.DataFrame({
            "trade_date": ["2020-01-03", "2020-01-03", "2020-01-10", "2020-01-10"],
            "stock_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "eligible": [True, True, True, True],
            "membership_effective_at": ["2019-01-01T00:00:00Z", "2019-01-01T00:00:00Z", "2020-01-10T02:00:00Z", "2019-01-01T00:00:00Z"],
        }).to_csv(root / "PIT_UNIVERSE_MEMBERSHIP.csv", index=False)
        write_sha_manifest(root, inputs)
        late_membership = invoke(root, root / "out_late_membership")
        receipt = json.loads((root / "out_late_membership" / "T2_EVENT_FREEZE_PRECONSUMPTION_AUDIT.json").read_text(encoding="utf-8"))
        if late_membership.returncode != 2 or "membership_not_effective_by_cutoff" not in receipt["failures"]:
            raise RuntimeError(f"expected late-membership failure: {late_membership.returncode} {receipt}")

        pd.DataFrame({
            "trade_date": ["2020-01-03", "2020-01-03", "2020-01-10", "2020-01-10"],
            "stock_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "eligible": [True, True, True, True],
            "membership_effective_at": ["2019-01-01T00:00:00Z"] * 4,
        }).to_csv(root / "PIT_UNIVERSE_MEMBERSHIP.csv", index=False)

        pd.DataFrame({
            "event_id": ["event-after-window"], "stock_code": ["000001.SZ"],
            "published_at_utc": ["2020-01-10T01:30:01Z"],
            "source_response_sha256": ["c" * 64], "source_url": ["https://example.invalid/event-after-window"],
        }).to_parquet(root / "STRICT_PIT_EVENTS.parquet", index=False)
        write_sha_manifest(root, inputs)
        late_event = invoke(root, root / "out_late_event")
        receipt = json.loads((root / "out_late_event" / "T2_EVENT_FREEZE_PRECONSUMPTION_AUDIT.json").read_text(encoding="utf-8"))
        if late_event.returncode != 2 or "event_after_maximum_origin_cutoff" not in receipt["failures"]:
            raise RuntimeError(f"expected late-event failure: {late_event.returncode} {receipt}")

        pd.DataFrame({
            "event_id": ["event-1"], "stock_code": ["000001.SZ"],
            "published_at_utc": ["2020-01-02T01:00:00Z"],
            "source_response_sha256": ["c" * 64], "source_url": ["https://example.invalid/event-1"],
        }).to_parquet(root / "STRICT_PIT_EVENTS.parquet", index=False)

        pd.DataFrame({
            "stock_code": ["000001.SZ"], "coverage_start_date": ["2019-01-01"], "coverage_end_date": ["2020-12-31"],
            "coverage_status": ["COVERED"], "source_system": ["test"], "source_snapshot_sha256": ["a" * 64],
        }).to_csv(root / "SOURCE_COVERAGE_RECEIPTS.csv", index=False)
        write_sha_manifest(root, inputs)
        failed = invoke(root, root / "out_fail")
        receipt = json.loads((root / "out_fail" / "T2_EVENT_FREEZE_PRECONSUMPTION_AUDIT.json").read_text(encoding="utf-8"))
        if failed.returncode != 2 or receipt["status"] != "FAIL_CLOSED_T2_EVENT_FREEZE_PRECONSUMPTION" or "eligible_universe_not_fully_covered" not in receipt["failures"]:
            raise RuntimeError(f"expected coverage failure: {failed.returncode} {receipt}")

        pd.DataFrame({
            "stock_code": ["000001.SZ", "000002.SZ"],
            "coverage_start_date": ["2019-01-01", "2019-01-01"],
            "coverage_end_date": ["2020-12-31", "2020-01-03"],
            "coverage_status": ["COVERED", "COVERED"],
            "source_system": ["test", "test"],
            "source_snapshot_sha256": ["a" * 64, "b" * 64],
        }).to_csv(root / "SOURCE_COVERAGE_RECEIPTS.csv", index=False)
        write_sha_manifest(root, inputs)
        stale = invoke(root, root / "out_stale")
        receipt = json.loads((root / "out_stale" / "T2_EVENT_FREEZE_PRECONSUMPTION_AUDIT.json").read_text(encoding="utf-8"))
        if stale.returncode != 2 or receipt["covered_eligible_membership_row_count"] != 3 or "eligible_universe_not_fully_covered" not in receipt["failures"]:
            raise RuntimeError(f"expected stale-coverage failure: {stale.returncode} {receipt}")
    print("PASS: label-free event freeze auditor accepts complete coverage and rejects missing or stale coverage")


if __name__ == "__main__":
    main()


