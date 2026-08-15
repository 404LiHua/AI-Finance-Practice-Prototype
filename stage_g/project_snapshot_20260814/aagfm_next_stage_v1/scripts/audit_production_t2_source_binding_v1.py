from __future__ import annotations

"""Fail-closed source-binding audit for the recovered production-T2 package.

This tool never opens a FRESH payload, trains a model, or invokes a materializer.
It validates only the user-provided source inbox, parses Python sources without
executing them, and replays the pure REV8 T2 target function against the already
authorized TRAIN target artifact.
"""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ENTRYPOINTS = (
    "src/rev8_targets.py",
    "scripts/materialize_rev8_selected_target.py",
    "scripts/materialize_rg3_features.py",
    "scripts/materialize_fresh1_confirmation_data.py",
    "scripts/materialize_fresh2_confirmation_data.py",
    "scripts/materialize_fresh3_incumbent_confirmation.py",
    "scripts/build_confirmed_safe_model.py",
)

REQUIRED_RUNTIME_MODULES = {
    "scripts/materialize_rg3_features.py": ("src.rg3_features",),
    "scripts/build_confirmed_safe_model.py": (
        "scripts.run_fresh1_one_shot_confirmation",
        "src.confirmed_safe_model",
        "src.rg2_calibrated_ordinal",
        "src.rg3_features",
    ),
}

REQUIRED_RAW_INPUTS = {
    "scripts/materialize_rev8_selected_target.py": (
        "data/rg1_4_materialized/samples.csv.gz",
    ),
    "scripts/materialize_rg3_features.py": (
        "data/rg1_4_materialized/weekly_panel.csv.gz",
        "data/rg3_materialized/raw 312 daily files (external DAILY_ROOT)",
        "structural panel (external PANEL_SOURCE)",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def parse_manifest(inbox: Path) -> list[dict[str, str]]:
    with (inbox / "SHA256_MANIFEST.csv").open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def manifest_audit(inbox: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    expected = {row["relative_path"]: row for row in parse_manifest(inbox)}
    actual_paths = sorted(
        path for path in inbox.rglob("*") if path.is_file() and path.name != "SHA256_MANIFEST.csv"
    )
    actual = {relative_posix(inbox, path): path for path in actual_paths}
    rows = []
    for relative in sorted(set(expected) | set(actual)):
        expected_row = expected.get(relative)
        path = actual.get(relative)
        size = path.stat().st_size if path else None
        digest = sha256(path) if path else None
        passed = bool(
            expected_row
            and path
            and int(expected_row["size_bytes"]) == size
            and expected_row["sha256"] == digest
        )
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": digest,
                "expected_size_bytes": int(expected_row["size_bytes"]) if expected_row else None,
                "expected_sha256": expected_row["sha256"] if expected_row else None,
                "manifest_pass": passed,
            }
        )
    summary = {
        "manifest_rows": len(expected),
        "actual_files": len(actual),
        "missing_files": sum(item["size_bytes"] is None for item in rows),
        "unexpected_files": sum(item["expected_size_bytes"] is None for item in rows),
        "passing_files": sum(item["manifest_pass"] for item in rows),
        "failing_files": sum(not item["manifest_pass"] for item in rows),
    }
    return rows, summary


def source_syntax_and_imports(inbox: Path) -> dict[str, object]:
    details: dict[str, object] = {}
    for path in sorted(inbox.rglob("*.py")):
        relative = relative_posix(inbox, path)
        source = path.read_text(encoding="utf-8")
        result: dict[str, object] = {"sha256": sha256(path), "syntax_pass": False, "imports": []}
        try:
            tree = ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            result["imports"] = sorted(set(imports))
            result["syntax_pass"] = True
        except (SyntaxError, UnicodeDecodeError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        details[relative] = result
    return details


def t2_core_contract(inbox: Path, train_target: Path) -> dict[str, object]:
    sys.dont_write_bytecode = True
    source = inbox / "src/rev8_targets.py"
    spec = importlib.util.spec_from_file_location("source_bound_rev8_targets", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load rev8 target source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = pd.read_csv(train_target, dtype={"stock_code": str})
    input_frame = observed[
        ["trade_date", "stock_code", "raw_target_return_h4", "raw_target_valid"]
    ].rename(
        columns={"raw_target_return_h4": "target_return_h4", "raw_target_valid": "target_valid"}
    )
    # T2 is independent of volatility; this required column is supplied only to
    # exercise the pure source function without fabricating any target value.
    input_frame["realized_volatility_8w"] = 0.1
    derived = module.build_target_variants(input_frame)
    market_equal = np.allclose(
        derived["market_h4_median"].to_numpy(float),
        observed["market_h4_median"].to_numpy(float),
        equal_nan=True,
        rtol=0.0,
        atol=1e-15,
    )
    return_equal = np.allclose(
        derived["T2_return"].to_numpy(float),
        observed["target_return_h4"].to_numpy(float),
        equal_nan=True,
        rtol=0.0,
        atol=1e-15,
    )
    threshold_equal = np.allclose(
        derived["T2_threshold"].to_numpy(float),
        observed["target_threshold"].to_numpy(float),
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    valid_equal = bool(
        (derived["T2_valid"].astype(bool).to_numpy() == observed["target_valid"].astype(bool).to_numpy()).all()
    )
    label_equal = bool(
        (derived["T2_label"].astype("Int64").fillna(-1).to_numpy() ==
         observed["ordinal_target"].astype("Int64").fillna(-1).to_numpy()).all()
    )
    return {
        "train_target_path": str(train_target),
        "train_target_sha256": sha256(train_target),
        "rows": int(len(observed)),
        "market_median_exact_to_1e15": bool(market_equal),
        "relative_return_exact_to_1e15": bool(return_equal),
        "threshold_exact": bool(threshold_equal),
        "validity_exact": valid_equal,
        "ordinal_label_exact": label_equal,
        "pass": bool(market_equal and return_equal and threshold_equal and valid_equal and label_equal),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", required=True, type=Path)
    parser.add_argument("--train-target", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    inbox = args.inbox.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)

    manifest_rows, manifest = manifest_audit(inbox)
    syntax = source_syntax_and_imports(inbox)
    entrypoints = {name: (inbox / name).is_file() for name in REQUIRED_ENTRYPOINTS}
    unresolved_modules = {
        script: [module for module in modules if not (inbox / (module.replace(".", "/") + ".py")).is_file()]
        for script, modules in REQUIRED_RUNTIME_MODULES.items()
    }
    unresolved_modules = {key: value for key, value in unresolved_modules.items() if value}
    missing_raw_inputs = {
        script: [item for item in items if not (inbox / item).is_file()]
        for script, items in REQUIRED_RAW_INPUTS.items()
    }
    t2 = t2_core_contract(inbox, args.train_target.resolve())
    all_syntax = all(bool(item["syntax_pass"]) for item in syntax.values())
    accepted = bool(
        manifest["failing_files"] == 0
        and all(entrypoints.values())
        and all_syntax
        and not unresolved_modules
        and not any(missing_raw_inputs.values())
        and t2["pass"]
    )
    status = (
        "PASS_READY_FOR_CONTROLLED_SOURCE_COPY" if accepted
        else "FAIL_CLOSED_INCOMPLETE_RUNTIME_AND_RAW_SOURCE_BINDING"
    )
    audit = {
        "node_id": "AA_GFMNET_PRODUCTION_T2_SOURCE_BINDING_V1",
        "status": status,
        "provenance": {
            "delivery": "user copied archive subset into workspace inbox",
            "direct_archive_inspection": False,
            "direct_archive_inspection_reason": "Codex external-directory approval backend returned 502",
            "original_archive_identity_independently_verified": False,
        },
        "manifest": manifest,
        "required_entrypoints": entrypoints,
        "python_source_syntax": syntax,
        "missing_runtime_modules": unresolved_modules,
        "missing_materializer_inputs": missing_raw_inputs,
        "t2_core_contract": t2,
        "fresh_payloads_opened": False,
        "fresh_labels_read": False,
        "screening_read": False,
        "final_read": False,
        "materializers_executed": False,
        "model_trained": False,
        "gpu_used": False,
        "controlled_source_copy_allowed": accepted,
        "production_replacement_allowed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output / "SOURCE_BINDING_MANIFEST_SHA256.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["relative_path", "size_bytes", "sha256", "expected_size_bytes", "expected_sha256", "manifest_pass"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output / "SOURCE_BINDING_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "node_id": audit["node_id"],
        "status": status,
        "source_files_read": manifest["actual_files"],
        "fresh_payloads_opened": False,
        "fresh_labels_read": False,
        "materializers_executed": False,
        "model_trained": False,
        "production_assets_modified": False,
        "controlled_source_copy_created": False,
        "created_at_utc": audit["created_at_utc"],
    }
    (output / "SOURCE_BINDING_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "output_root": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


