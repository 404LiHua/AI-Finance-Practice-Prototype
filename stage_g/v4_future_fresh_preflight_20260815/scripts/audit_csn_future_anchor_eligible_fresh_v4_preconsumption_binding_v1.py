from __future__ import annotations

"""V4-only label-free binding audit before a custodian may access labels.

The auditor has no label-file argument.  It binds only feature-side inputs, two
sealed prediction files, immutable model identities, and the label hash declared
by the materialization receipt.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ORIGINS = ("2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07")
MANIFEST_SHA = "c7ee9368bf0e71eb21efc3f4de05b86f0a68335008891afc401e8a7b0fa6908e"
SPEC_SHA = "49b4e6d6c441bb44e94949da35a86d4bc2fb2324c8952394b99bb4c5a6786741"
KEYS = ("origin_date", "stock_code")
PREDICTION_FIELDS = ("h4_prediction", "p_down", "p_neutral", "p_up")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_contract(path: Path, universe: pd.DataFrame, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path, columns=[*KEYS, *PREDICTION_FIELDS])
    except Exception as error:
        raise RuntimeError(f"FAIL_CLOSED_V4_{label}_PREDICTION_SCHEMA") from error
    frame.origin_date = pd.to_datetime(frame.origin_date, errors="coerce").dt.date.astype(str)
    frame.stock_code = frame.stock_code.astype(str).str.upper()
    values = frame.loc[:, PREDICTION_FIELDS].to_numpy(dtype=float)
    probability = frame.loc[:, ("p_down", "p_neutral", "p_up")].to_numpy(dtype=float)
    expected = pd.MultiIndex.from_frame(universe.loc[:, KEYS])
    actual = pd.MultiIndex.from_frame(frame.loc[:, KEYS])
    if frame.origin_date.isna().any() or frame.duplicated(list(KEYS)).any() or not actual.sort_values().equals(expected.sort_values()):
        raise RuntimeError(f"FAIL_CLOSED_V4_{label}_PREDICTION_KEY_DOMAIN")
    if not np.isfinite(values).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError(f"FAIL_CLOSED_V4_{label}_PREDICTION_NUMERIC")
    return frame.sort_values(list(KEYS)).reset_index(drop=True)


def universe_contract(path: Path) -> pd.DataFrame:
    universe = pd.read_parquet(path, columns=list(KEYS)).copy()
    universe.origin_date = pd.to_datetime(universe.origin_date, errors="coerce").dt.date.astype(str)
    universe.stock_code = universe.stock_code.astype(str).str.upper()
    if universe.origin_date.isna().any() or universe.duplicated(list(KEYS)).any() or tuple(sorted(universe.origin_date.unique())) != ORIGINS:
        raise RuntimeError("FAIL_CLOSED_V4_BINDING_UNIVERSE_ORIGINS")
    counts = universe.groupby("origin_date").stock_code.nunique()
    if len(counts) != 8 or counts.nunique() != 1 or not 200 <= int(counts.iloc[0]) <= 300:
        raise RuntimeError("FAIL_CLOSED_V4_BINDING_UNIVERSE_STOCKS")
    return universe.sort_values(list(KEYS)).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    for option in ("--candidate-predictions", "--anchor-predictions", "--candidate-receipt", "--anchor-receipt", "--v4-input-contract", "--materialization-receipt", "--universe", "--candidate-manifest", "--candidate-specification", "--anchor-model", "--output"):
        parser.add_argument(option, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("FAIL_CLOSED_V4_BINDING_OUTPUT_EXISTS")
    universe = universe_contract(args.universe)
    candidate = prediction_contract(args.candidate_predictions, universe, "CANDIDATE")
    anchor = prediction_contract(args.anchor_predictions, universe, "ANCHOR")
    candidate_receipt, anchor_receipt = read_json(args.candidate_receipt), read_json(args.anchor_receipt)
    contract, materialization, anchor_model = read_json(args.v4_input_contract), read_json(args.materialization_receipt), read_json(args.anchor_model)
    input_hashes = materialization.get("output_sha256", {})
    contract_hashes = contract.get("input_sha256", {})
    checks = {
        "v4_input_contract_pass": contract.get("status") == "PASS_V4_LABEL_FREE_INPUT_CONTRACT" and contract.get("origin_dates") == list(ORIGINS),
        "materialization_pass_without_label_read": materialization.get("status") == "PASS_V4_SEALED_INPUT_MATERIALIZATION" and materialization.get("labels_read") is False and materialization.get("labels_opened_by_materialization") is False,
        "candidate_prediction_receipt_pass": candidate_receipt.get("status") == "PASS_LABEL_FREE_CANDIDATE_BATCH_PREDICTION",
        "anchor_prediction_receipt_pass": anchor_receipt.get("status") == "PASS_LABEL_FREE_ANCHOR_PREDICTION",
        "candidate_immutable_identity": sha256(args.candidate_manifest) == MANIFEST_SHA and sha256(args.candidate_specification) == SPEC_SHA and candidate_receipt.get("candidate_id") == "AA_GFMNET_CROSS_SECTIONAL_NEUTRALIZED_RESIDUAL_TCN_V1" and candidate_receipt.get("model_manifest_sha256") == MANIFEST_SHA and candidate_receipt.get("model_specification_sha256") == SPEC_SHA,
        "anchor_immutable_identity": anchor_model.get("model_id") == "RG_OBGNET_CONFIRMED_SAFE_V1_1" and anchor_receipt.get("anchor_kernel_id") == "RG_OBGNET_CONFIRMED_SAFE_V1_1" and anchor_receipt.get("model_sha256") == sha256(args.anchor_model),
        "candidate_input_hashes_match": candidate_receipt.get("numeric_sha256") == input_hashes.get("FRESH_NUMERIC.npz") and candidate_receipt.get("technical_sha256") == input_hashes.get("FRESH_TECHNICAL.parquet") and candidate_receipt.get("fundamentals_sha256") == input_hashes.get("FRESH_FUNDAMENTALS.parquet"),
        "anchor_universe_hash_matches": anchor_receipt.get("universe_sha256") == sha256(args.universe),
        "input_contract_hashes_match_materialization": all(contract_hashes.get(name) == input_hashes.get(name) for name in ("FRESH_NUMERIC.npz", "FRESH_TECHNICAL.parquet", "FRESH_FUNDAMENTALS.parquet", "FRESH_UNIVERSE.parquet")) and contract_hashes.get("MATERIALIZATION_RECEIPT.json") == sha256(args.materialization_receipt),
        "candidate_anchor_prediction_keys_equal": candidate.loc[:, KEYS].equals(anchor.loc[:, KEYS]),
        "exact_eight_pre_registered_origins": tuple(sorted(candidate.origin_date.unique())) == ORIGINS,
        "sealed_label_hash_declared_only": bool(input_hashes.get("SEALED_FRESH_H4_LABELS.parquet")),
        "no_prediction_label_or_return_read": all(receipt.get("labels_read") is False and receipt.get("returns_read") is False and receipt.get("fresh_labels_read") is False for receipt in (candidate_receipt, anchor_receipt)),
        "resource_schedule_receipted": candidate_receipt.get("gpu_jobs_concurrent") in (0, 1) and int(anchor_receipt.get("workers", 1)) in (1, 2),
    }
    status = "PASS_V4_PRECONSUMPTION_PREDICTION_BINDING_READY_FOR_CUSTODIAN" if all(checks.values()) else "FAIL_V4_PRECONSUMPTION_PREDICTION_BINDING"
    receipt = {"node_id": "AA_GFMNET_CSN_FUTURE_ANCHOR_ELIGIBLE_FRESH_V4_PRECONSUMPTION_BINDING_V1", "status": status, "checks": checks, "origin_dates": list(ORIGINS), "stock_count": int(universe.stock_code.nunique()), "rows": len(universe), "candidate_prediction_sha256": sha256(args.candidate_predictions), "anchor_prediction_sha256": sha256(args.anchor_predictions), "candidate_manifest_sha256": sha256(args.candidate_manifest), "candidate_specification_sha256": sha256(args.candidate_specification), "anchor_model_sha256": sha256(args.anchor_model), "sealed_labels_sha256_declared_without_read": input_hashes.get("SEALED_FRESH_H4_LABELS.parquet"), "labels_read": False, "fresh_labels_read": False, "returns_read": False, "production_kernel_modified": False, "gpu_jobs_concurrent": 0, "cpu_thread_cap": 1, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if status.startswith("FAIL"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
