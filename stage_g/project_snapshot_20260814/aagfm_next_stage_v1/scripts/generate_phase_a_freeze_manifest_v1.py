"""Generate Phase A freeze artifacts for the AA-GFMNet next-stage track.

This script is intentionally CPU-only.  It binds known strict-PIT assets,
records gaps that must not be guessed, and freezes the module routing for the
next four-trading-day H4 research work only.  It does not read SCREENING, FINAL, sealed
holdouts, or train any model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def file_row(root: Path, role: str, path: Path, required: bool, note: str) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "role": role,
        "path": rel(path, root),
        "required": required,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
        "note": note,
    }


def inspect_sequence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    with np.load(path, allow_pickle=False) as data:
        x_shape = list(data["x"].shape) if "x" in data.files else None
        origin_count = int(len(data["origin_dates"])) if "origin_dates" in data.files else None
        stock_count = int(len(data["stock_codes"])) if "stock_codes" in data.files else None
        feature_names = data["feature_names"].astype(str).tolist() if "feature_names" in data.files else []
        future_rows = None
        if {"origin_dates", "source_trade_date"}.issubset(data.files):
            origin_dates = data["origin_dates"].astype("datetime64[D]")
            source_dates = data["source_trade_date"].astype("datetime64[D]")
            future = (~np.isnat(source_dates)) & (source_dates >= origin_dates[:, None, None])
            future_rows = int(future.sum())
        node_available = int(data["node_available"].sum()) if "node_available" in data.files else None
    return {
        "exists": True,
        "x_shape": x_shape,
        "origin_count": origin_count,
        "stock_count": stock_count,
        "feature_names": feature_names,
        "future_rows": future_rows,
        "node_available": node_available,
    }


def inspect_labels(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    frame = pd.read_parquet(
        path,
        columns=["origin_id", "stock_code", "h4_return", "label_valid", "target_horizon_trading_days"],
    )
    return {
        "exists": True,
        "rows": int(len(frame)),
        "unique_keys": int(frame[["origin_id", "stock_code"]].drop_duplicates().shape[0]),
        "valid_labels": int(frame["label_valid"].fillna(False).astype(bool).sum()),
        "finite_h4": int(np.isfinite(frame["h4_return"].to_numpy(dtype=float)).sum()),
        "target_horizon_trading_days": sorted(
            pd.to_numeric(frame["target_horizon_trading_days"], errors="coerce").dropna().astype(int).unique().tolist()
        ),
    }


def inspect_oof(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    frame = pd.read_parquet(path)
    id_column = "model_id" if "model_id" in frame.columns else "candidate_id" if "candidate_id" in frame.columns else None
    return {
        "exists": True,
        "rows": int(len(frame)),
        "unique_keys": int(frame[["origin_id", "stock_code"]].drop_duplicates().shape[0]),
        "prediction_valid_rows": int(frame["prediction_valid"].fillna(False).astype(bool).sum()),
        "id_column": id_column,
        "ids": sorted(frame[id_column].astype(str).unique().tolist()) if id_column else [],
    }


def inspect_csv_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="Workspace project root containing AI_Finance_Prototype and deliverables.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "audits" / "phase_a_freeze_manifest_v1",
    )
    args = parser.parse_args()
    ai_root = args.ai_root.resolve()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    pit = ai_root / "research_tracks" / "pit_information_incremental_v1"
    next_stage = ai_root / "research_tracks" / "aagfm_next_stage_v1"
    ap_tdsf = ai_root / "research_tracks" / "ap_tdsf_anchor_preserving_v1"
    assets = {
        "work_package_06": next_stage / "WORK_PACKAGE_06_MODEL_STRENGTHENING_AND_RESOURCE_BALANCE.md",
        "formal_stock_universe_300": pit / "frozen_inputs" / "m5_frozen_tensor_300stocks_2023_2025_v3" / "M5_FORMAL_STOCK_UNIVERSE_300.csv",
        "strict_pit_technical": pit / "sources" / "technical_textcu_v2_strict_pit_20260812_v1" / "TECHNICAL_ASOF_FEATURES_STRICT_PIT.parquet",
        "strict_pit_fundamentals": pit / "sources" / "fundamentals_textcu_v2_strict_pit_20260812_v3" / "CSMAR_PIT_FUNDAMENTALS_ASOF_STRICT_PIT.parquet",
        "sequence8_numeric_tensor": pit / "sources" / "incumbent_numeric_6d_textcu_v7_sequence8_pit_20260813_v1" / "TEXTCU_V5_INCUMBENT_NUMERIC_6D_SEQUENCE8_PIT_TENSOR.npz",
        "sequence8_build_receipt": pit / "sources" / "incumbent_numeric_6d_textcu_v7_sequence8_pit_20260813_v1" / "BUILD_RECEIPT.json",
        "h4_labels": pit / "sources" / "h4_labels_textcu_v2_20260812_run4" / "H4_LABELS_TEXTCU_V2_337X300.parquet",
        "h4_label_receipt": pit / "sources" / "h4_labels_textcu_v2_20260812_run4" / "LABEL_BUILD_RECEIPT.json",
        "bind_receipt": pit / "sources" / "incumbent_architecture_recomputation_bind_20260813_v4_seq8" / "BIND_RECEIPT.json",
        "origin_registry": pit / "sources" / "incumbent_architecture_recomputation_bind_20260813_v4_seq8" / "ORIGIN_REGISTRY_337.csv",
        "split_registry": pit / "sources" / "incumbent_architecture_recomputation_bind_20260813_v4_seq8" / "SPLIT_REGISTRY.csv",
        "industry_adjacency": pit / "sources" / "incumbent_architecture_recomputation_bind_20260813_v4_seq8" / "INDUSTRY_ADJACENCY_300X300.npy",
        "architecture_model_oof": pit / "outputs" / "textcu_architecture_recomputation_seq8_20260813_closed_v4" / "INCUMBENT_ARCH_RECOMPUTATION_OOF.parquet",
        "architecture_naive_oof": pit / "outputs" / "textcu_architecture_recomputation_seq8_20260813_closed_v4" / "NAIVE_OOF.parquet",
        "architecture_key_audit": pit / "outputs" / "textcu_architecture_recomputation_seq8_20260813_closed_v4" / "OOF_KEY_AUDIT.csv",
        "architecture_result_report": pit / "governance" / "TEXTCU_ARCHITECTURE_RECOMPUTATION_FORMAL_RESULT_REPORT_20260813.json",
        "text_rebinding_receipt": pit / "sources" / "textcu_current_rebind_20260813_v2" / "REBINDING_RECEIPT.json",
        "text_event_panel": pit / "sources" / "textcu_current_rebind_20260813_v2" / "CURRENT_TEXT_EVENT_PANEL_337X300.parquet",
        "text_visible_event_table": pit / "sources" / "textcu_current_rebind_20260813_v2" / "CURRENT_TEXT_VISIBLE_EVENT_TABLE.parquet",
        "text_signal_report": pit / "governance" / "TEXTCU_R1_TEXT_ONLY_SIGNAL_FORMAL_REPORT_20260813.json",
        "title_body_hash_ngram_oof": pit / "outputs" / "textcu_current_text_signal_audit_20260813_v4" / "TITLE_BODY_HASH_NGRAM_OOF.parquet",
        "text_signal_gate_decision": pit / "outputs" / "textcu_current_text_signal_audit_20260813_v4" / "TEXT_SIGNAL_GATE_DECISION.csv",
        "r26_closure": ap_tdsf / "candidate_freeze" / "r26_candidate_freeze_v1" / "R26_CLOSURE_REPORT.md",
        "production_model_json": project_root / "deliverables" / "RG_OBGNet_source_v1" / "models" / "rg_obgnet_confirmed_safe_v1_1" / "MODEL.json",
        "production_inference_code": project_root / "deliverables" / "RG_OBGNet_source_v1" / "src" / "confirmed_safe_model.py",
        "production_feature_code": project_root / "deliverables" / "RG_OBGNet_source_v1" / "src" / "rg3_features.py",
    }

    manifest_rows = [
        file_row(project_root, role, path, True, "phase_a_bound_input")
        for role, path in assets.items()
    ]
    pd.DataFrame(manifest_rows).to_csv(output_root / "PHASE_A_FREEZE_MANIFEST.csv", index=False, encoding="utf-8")

    sequence_info = inspect_sequence(assets["sequence8_numeric_tensor"])
    label_info = inspect_labels(assets["h4_labels"])
    model_oof_info = inspect_oof(assets["architecture_model_oof"])
    title_oof_info = inspect_oof(assets["title_body_hash_ngram_oof"])
    formal_stock_rows = inspect_csv_rows(assets["formal_stock_universe_300"])
    origin_rows = inspect_csv_rows(assets["origin_registry"])
    split_rows = inspect_csv_rows(assets["split_registry"])

    gaps = [
        {
            "gap_id": "T2_TARGET_SEMANTICS",
            "status": "PARTIALLY_RECOVERED_NOT_RECONSTRUCTABLE",
            "need": "Production target is known as T2_MARKET_RELATIVE_FIXED, but its executable full-universe/split/filtering construction has a separate reconstruction track and cannot be inferred from this four-trading-day H4 label.",
            "allowed_acquisition": "Use only the production-T2 reconstruction package and frozen source artifacts with paths/SHA-256; do not derive it from this label or the target name.",
        },
        {
            "gap_id": "H4_TO_T2_DERIVATION",
            "status": "TARGET_HORIZON_CONFLICT_DERIVATION_PROHIBITED",
            "need": "Bound H4 labels are four-trading-day absolute returns. Production T2 is four-week market-relative three-class. The two targets cannot be mapped or compared as the same target.",
            "allowed_acquisition": "Do not derive a production T2 head from this H4 label. Use the separate production-T2 reconstruction track after its source and authorization gates pass.",
        },
        {
            "gap_id": "RELIABILITY_SEMANTICS",
            "status": "BOUND_FROM_PRODUCTION_MODEL_JSON" if assets["production_model_json"].is_file() else "MISSING_NOT_GUESSABLE",
            "need": "Freeze reliability output formula and calibration protocol.",
            "allowed_acquisition": "Use operational_reliability in the confirmed production MODEL.json; do not invent a new production formula.",
        },
        {
            "gap_id": "FROZEN_CN_SENTENCE_EMBEDDING_MODEL",
            "status": "MISSING_OPTIONAL_ROUTE_CLOSED",
            "need": "Local frozen model, tokenizer, version, and SHA-256 before sentence embedding route can run.",
            "allowed_acquisition": "Provide the local model package and manifest; no downloading or ad hoc replacement in this phase.",
        },
        {
            "gap_id": "PRODUCTION_MODEL_JSON",
            "status": "BOUND_FROM_DELIVERABLES" if assets["production_model_json"].is_file() else "MISSING_NOT_GUESSABLE",
            "need": "Production anchor MODEL.json for production-comparable T2 checks.",
            "allowed_acquisition": "Bound from deliverables/RG_OBGNet_source_v1/models/rg_obgnet_confirmed_safe_v1_1/MODEL.json.",
        },
    ]
    with (output_root / "PHASE_A_GAP_REGISTER.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gaps[0].keys()))
        writer.writeheader()
        writer.writerows(gaps)

    module_rows = [
        {
            "module_id": "NUMERIC_SEQUENCE8_H4_BACKBONE",
            "status": "ALLOWED_H4_RESEARCH_ONLY",
            "reason": "Strict-PIT sequence8 numeric tensor and four-trading-day absolute H4 labels are bound.",
            "next_action": "Build a single-task four-trading-day H4 research candidate with dry-run first. No production-T2-comparable or replacement claim is allowed.",
        },
        {
            "module_id": "TITLE_BODY_HASH_NGRAM",
            "status": "ALLOWED_RESTRICTED_RESIDUAL_ONLY",
            "reason": "R1 text-only signal gate passed.",
            "next_action": "Use only as small auditable residual/gate source; do not replace numeric backbone.",
        },
        {
            "module_id": "TFIDF_BODY_NGRAM",
            "status": "CLOSED_NEGATIVE",
            "reason": "R1 failed the positive-block gate.",
            "next_action": "Keep as negative evidence; no tuning or fusion.",
        },
        {
            "module_id": "FROZEN_CN_SENTENCE_EMBEDDING",
            "status": "CLOSED_MISSING_LOCAL_MODEL",
            "reason": "No local frozen model and SHA-256 are bound.",
            "next_action": "Do not execute until model package is provided and frozen.",
        },
        {
            "module_id": "R26_LOW_GAMMA_XGB_ANCHOR_BLEND",
            "status": "CLOSED_FROZEN_TEST_FAIL",
            "reason": "R26 closure report says promotion_allowed=false.",
            "next_action": "Do not tune gamma/window/seed.",
        },
        {
            "module_id": "PRODUCTION_T2_KERNEL",
            "status": "PRESERVE_CURRENT",
            "reason": "Current production kernel cannot be replaced without restored T2 semantics and full gates.",
            "next_action": "Keep RG_OBGNET_CONFIRMED_SAFE_V1_1 unchanged.",
        },
    ]
    with (output_root / "PHASE_A_MODULE_ROUTING.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(module_rows[0].keys()))
        writer.writeheader()
        writer.writerows(module_rows)

    resource_plan = {
        "status": "FROZEN",
        "cpu_roles": ["file_io", "sha256", "manifest", "csv_parquet_summary", "key_audit", "reporting"],
        "gpu_roles": ["training", "validation_inference", "checkpoint_reload_predict"],
        "scheduling_rules": [
            "one heavy GPU training job at a time",
            "run CPU audits between folds",
            "fail closed before GPU consumption if dry-run fails",
            "do not read SCREENING, FINAL, sealed holdout, or future external data in Phase A",
        ],
    }
    write_json(output_root / "PHASE_A_RESOURCE_PLAN.json", resource_plan)

    hard_failures = []
    if formal_stock_rows != 300:
        hard_failures.append(f"formal_stock_rows={formal_stock_rows}")
    if origin_rows != 337:
        hard_failures.append(f"origin_rows={origin_rows}")
    if split_rows != 6:
        hard_failures.append(f"split_rows={split_rows}")
    if sequence_info.get("x_shape") != [337, 8, 300, 6]:
        hard_failures.append(f"sequence_shape={sequence_info.get('x_shape')}")
    if sequence_info.get("future_rows") not in (0, None):
        hard_failures.append(f"sequence_future_rows={sequence_info.get('future_rows')}")
    if label_info.get("rows") != 101100:
        hard_failures.append(f"label_rows={label_info.get('rows')}")
    if label_info.get("target_horizon_trading_days") != [4]:
        hard_failures.append(f"label_horizon={label_info.get('target_horizon_trading_days')}")
    if model_oof_info.get("unique_keys") != 101100:
        hard_failures.append(f"model_oof_unique_keys={model_oof_info.get('unique_keys')}")
    if title_oof_info.get("unique_keys") != 101100:
        hard_failures.append(f"title_oof_unique_keys={title_oof_info.get('unique_keys')}")
    missing_required = [row["role"] for row in manifest_rows if row["required"] and not row["exists"]]
    if missing_required:
        hard_failures.append(f"missing_required={missing_required}")

    decision = {
        "node_id": "AA_GFMNET_PHASE_A_FREEZE_MANIFEST_V2",
        "status": "PASS_PHASE_A_READY_FOR_RESEARCH_H4_PACKAGE_ONLY" if not hard_failures else "FAIL_CLOSED_PHASE_A",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ai_root": str(ai_root),
        "hard_failures": hard_failures,
        "inspections": {
            "formal_stock_rows": formal_stock_rows,
            "origin_rows": origin_rows,
            "split_rows": split_rows,
            "sequence": sequence_info,
            "labels": label_info,
            "architecture_model_oof": model_oof_info,
            "title_body_hash_ngram_oof": title_oof_info,
        },
        "gaps_not_guessable": [
            gap["gap_id"]
            for gap in gaps
            if gap["status"] in {
                "MISSING_NOT_GUESSABLE",
                "PARTIALLY_RECOVERED_NOT_RECONSTRUCTABLE",
                "TARGET_HORIZON_CONFLICT_DERIVATION_PROHIBITED",
                "MISSING_OPTIONAL_ROUTE_CLOSED",
            }
        ],
        "screening_read": False,
        "final_read": False,
        "sealed_holdout_read": False,
        "model_trained": False,
        "gpu_used": False,
        "promotion_allowed": False,
    }
    write_json(output_root / "PHASE_A_DECISION.json", decision)
    manifest = [
        file_row(output_root, "phase_a_artifact", path, True, "phase_a_output")
        for path in sorted(output_root.glob("PHASE_A_*"))
        if path.name != "PHASE_A_SHA256_MANIFEST.csv"
    ]
    pd.DataFrame(manifest).to_csv(output_root / "PHASE_A_SHA256_MANIFEST.csv", index=False, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if hard_failures:
        sys.exit(2)


if __name__ == "__main__":
    main()


