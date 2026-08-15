from __future__ import annotations

"""CPU-only, fail-closed reconstruction test for production-T2 source binding.

The original archive scripts are loaded without modification.  Their output paths
and the relocated daily-source root are redirected to a new audit directory.  No
FRESH payload is opened and the safe-model builder is deliberately not executed.
"""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPECTED = {
    "samples": "60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6",
    "train_target": "aadada6cbdcaaefd0edd0df1a66b176daba6b015cee6d9f6867cf95d6d92204c",
    "rg3_features": "04f6b11b7296aa1d92bdc0a97d652565672ac90b1a01601b67f9a629989a9525",
    "panel": "2e5ce4511381005c33eea793bfb46f1ee009a5dc0f0cb1924afd8e4c4196c2fe",
    "daily_registry": "74633363e3313d62c38f9cf0fe3f641980133dbf22d87433fcc8f712431bf66f",
}

FRESH_READING_SCRIPTS = {
    "scripts/materialize_fresh1_confirmation_data.py",
    "scripts/materialize_fresh2_confirmation_data.py",
    "scripts/materialize_fresh3_incumbent_confirmation.py",
    "scripts/run_fresh1_one_shot_confirmation.py",
    "scripts/build_confirmed_safe_model.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inbox_manifest(inbox: Path) -> dict[str, object]:
    with (inbox / "SHA256_MANIFEST.csv").open("r", encoding="utf-8", newline="") as stream:
        expected = list(csv.DictReader(stream))
    actual = {
        path.relative_to(inbox).as_posix(): path
        for path in inbox.rglob("*")
        if path.is_file() and path.name != "SHA256_MANIFEST.csv"
    }
    problems = []
    for row in expected:
        path = actual.pop(row["relative_path"], None)
        if path is None:
            problems.append({"path": row["relative_path"], "problem": "missing"})
        elif path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            problems.append({"path": row["relative_path"], "problem": "hash_or_size_mismatch"})
    for relative in sorted(actual):
        problems.append({"path": relative, "problem": "not_listed_in_manifest"})
    return {
        "manifest_rows": len(expected),
        "actual_files": len(expected) + len(actual) - sum(item["problem"] == "missing" for item in problems),
        "problems": problems,
        "pass": not problems,
    }


def static_python_audit(inbox: Path) -> dict[str, object]:
    results = {}
    for path in sorted(inbox.rglob("*.py")):
        relative = path.relative_to(inbox).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
            results[relative] = {"syntax_pass": True, "sha256": sha256(path)}
        except (SyntaxError, UnicodeDecodeError) as error:
            results[relative] = {"syntax_pass": False, "error": f"{type(error).__name__}: {error}"}
    return {"files": results, "pass": all(item["syntax_pass"] for item in results.values())}


def verify_daily_mirror(inbox: Path) -> dict[str, object]:
    registry = pd.read_csv(inbox / "data/rg3_materialized/SOURCE_FILE_REGISTRY.csv", dtype={"stock_code": str})
    daily_root = inbox / "data/rg3_daily_raw"
    mismatches = []
    payload_rows = []
    for row in registry.sort_values("stock_code", kind="mergesort").itertuples(index=False):
        path = daily_root / f"{row.stock_code}.csv"
        if not path.is_file():
            mismatches.append({"stock_code": row.stock_code, "problem": "missing"})
            continue
        size = path.stat().st_size
        digest = sha256(path)
        payload_rows.append(f"{row.stock_code},{digest},{size}")
        if size != int(row.bytes) or digest != row.sha256:
            mismatches.append({"stock_code": row.stock_code, "problem": "hash_or_size_mismatch"})
    registry_digest = hashlib.sha256("\n".join(payload_rows).encode("utf-8")).hexdigest()
    extra_codes = sorted(
        path.stem for path in daily_root.glob("*.csv")
        if path.name != "SUPPLEMENT_DAYLY_VERIFICATION.csv" and path.stem not in set(registry.stock_code)
    )
    return {
        "registered_files": int(len(registry)),
        "mismatches": mismatches,
        "extra_codes": extra_codes,
        "computed_registry_sha256": registry_digest,
        "expected_registry_sha256": EXPECTED["daily_registry"],
        "pass": not mismatches and not extra_codes and registry_digest == EXPECTED["daily_registry"],
    }


def inspect_sample_scope(samples: Path) -> dict[str, object]:
    meta = pd.read_csv(samples, usecols=["fold_id", "split_role", "trade_date"], dtype={"fold_id": str, "split_role": str})
    roles = sorted(meta["split_role"].dropna().unique().tolist())
    forbidden = sorted(set(roles) & {"SCREENING", "FINAL", "SEALED_HOLDOUT", "FRESH"})
    return {
        "rows": int(len(meta)),
        "fold_ids": sorted(meta["fold_id"].dropna().unique().tolist()),
        "split_roles": roles,
        "forbidden_roles_present": forbidden,
        "pass": not forbidden,
    }


def load_source_module(name: str, path: Path, inbox: Path):
    sys.dont_write_bytecode = True
    if str(inbox) not in sys.path:
        sys.path.insert(0, str(inbox))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_target_dry_run(inbox: Path, output: Path) -> dict[str, object]:
    module = load_source_module(
        "source_binding_materialize_rev8", inbox / "scripts/materialize_rev8_selected_target.py", inbox
    )
    module.OUTPUT = output
    module.TARGET = output / "rev8_ro01_train_target.csv.gz"
    module.main()
    digest = sha256(module.TARGET)
    return {"output": str(module.TARGET), "sha256": digest, "expected_sha256": EXPECTED["train_target"], "pass": digest == EXPECTED["train_target"]}


def run_rg3_dry_run(inbox: Path, daily_root: Path, panel: Path, output: Path) -> dict[str, object]:
    module = load_source_module(
        "source_binding_materialize_rg3", inbox / "scripts/materialize_rg3_features.py", inbox
    )
    module.DAILY_ROOT = daily_root
    module.PANEL_SOURCE = panel
    module.WEEKLY = inbox / "data/rg1_4_materialized/weekly_panel.csv.gz"
    module.OUTPUT = output
    # The archive script writes a manifest with paths relative to ROOT.  In this
    # dry-run OUTPUT intentionally lives in a new audit folder, so bind only
    # that reporting root to the output parent; all input constants remain the
    # immutable inbox paths above.
    module.ROOT = output.parent
    module.main()
    feature_path = output / "rg3_features.csv.gz"
    digest = sha256(feature_path)
    return {"output": str(feature_path), "sha256": digest, "expected_sha256": EXPECTED["rg3_features"], "pass": digest == EXPECTED["rg3_features"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", required=True, type=Path)
    parser.add_argument("--panel-source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    inbox = args.inbox.resolve()
    panel = args.panel_source.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite audit output: {output}")
    output.mkdir(parents=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    manifest = verify_inbox_manifest(inbox)
    python = static_python_audit(inbox)
    samples = inbox / "data/rg1_4_materialized/samples.csv.gz"
    weekly = inbox / "data/rg1_4_materialized/weekly_panel.csv.gz"
    daily = verify_daily_mirror(inbox)
    sample_scope = inspect_sample_scope(samples)
    preconditions = {
        "inbox_manifest": manifest["pass"],
        "python_syntax": python["pass"],
        "samples_sha256": sha256(samples) == EXPECTED["samples"],
        "weekly_panel_present": weekly.is_file(),
        "daily_mirror": daily["pass"],
        "panel_sha256": sha256(panel) == EXPECTED["panel"],
        "sample_scope": sample_scope["pass"],
    }
    if not all(preconditions.values()):
        audit = {"status": "FAIL_CLOSED_PRECONDITION", "preconditions": preconditions, "manifest": manifest, "python": python, "daily": daily, "sample_scope": sample_scope}
    else:
        target = run_target_dry_run(inbox, output / "rev8_dry_run")
        rg3 = run_rg3_dry_run(inbox, inbox / "data/rg3_daily_raw", panel, output / "rg3_dry_run")
        status = "PASS_REPRODUCIBLE_TRAIN_AND_RG3_SOURCE_BINDING_FRESH_SEALED" if target["pass"] and rg3["pass"] else "FAIL_CLOSED_OUTPUT_IDENTITY"
        audit = {"status": status, "preconditions": preconditions, "manifest": manifest, "python": python, "daily": daily, "sample_scope": sample_scope, "rev8_dry_run": target, "rg3_dry_run": rg3}
    audit.update({
        "node_id": "AA_GFMNET_PRODUCTION_T2_SOURCE_BINDING_CPU_DRY_RUN_V1",
        "fresh_payloads_opened": False,
        "fresh_labels_read": False,
        "fresh_reading_scripts_executed": False,
        "fresh_reading_scripts_static_only": sorted(FRESH_READING_SCRIPTS),
        "screening_read": False,
        "final_read": False,
        "model_trained": False,
        "gpu_used": False,
        "production_assets_modified": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output / "SOURCE_BINDING_CPU_DRY_RUN_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {key: audit[key] for key in ("node_id", "status", "fresh_payloads_opened", "fresh_labels_read", "model_trained", "gpu_used", "production_assets_modified", "created_at_utc")}
    (output / "SOURCE_BINDING_CPU_DRY_RUN_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "output_root": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


