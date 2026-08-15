"""One-shot application of frozen E-6.1 gates to the eight existing candidate models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.e6.gates import apply_candidate_gates
from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


TABLE_FILES = {
    "overall": "overall_pooled_metrics.csv",
    "worst_fold": "worst_fold_summary.csv",
    "per_stock": "diagnostics_per_stock.csv",
    "industry": "diagnostics_industry.csv",
    "market_cap": "diagnostics_market_cap.csv",
    "return_decile": "diagnostics_return_decile.csv",
    "seed_summary": "seed_summary.csv",
    "pairwise_seed": "pairwise_seed_stability.csv",
    "seed_dispersion": "seed_prediction_dispersion.csv",
    "cost": "engineering_cost_summary.csv",
}


def _artifact_hash_key(filename: str) -> str:
    return f"{Path(filename).stem}_sha256"


def run(application_config_path: Path) -> Path:
    application = json.loads(application_config_path.read_text(encoding="utf-8"))
    if application["status"] != "ONE_SHOT_AUTHORIZED_NOT_YET_CONSUMED":
        raise ValueError("E-6 candidate application is not in unconsumed authorized state")
    output_root = resolve(application["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("one-shot E-6 candidate diagnostic read has already been consumed")
    gate_path = resolve(application["gate_config"])
    receipt_path = resolve(application["gate_freeze_receipt"])
    if sha256_file(gate_path) != application["gate_config_sha256"]:
        raise RuntimeError("frozen E-6 gate config hash mismatch")
    if sha256_file(receipt_path) != application["gate_freeze_receipt_sha256"]:
        raise RuntimeError("E-6 freeze receipt hash mismatch")
    if sha256_file(resolve(application["gate_implementation"])) != application["gate_implementation_sha256"]:
        raise RuntimeError("E-6 gate implementation hash mismatch")
    if sha256_file(Path(__file__).resolve()) != application["one_shot_executor_sha256"]:
        raise RuntimeError("E-6 one-shot executor hash mismatch")
    gate_config = json.loads(gate_path.read_text(encoding="utf-8"))
    freeze_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not freeze_receipt["passed"] or freeze_receipt["candidate_metrics_read"]:
        raise RuntimeError("candidate gate freeze receipt is not clean")
    diagnostic_source = gate_config["source_diagnostics"]
    metadata_path = resolve(diagnostic_source["metadata"])
    acceptance_path = resolve(diagnostic_source["acceptance"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if sha256_file(metadata_path) != diagnostic_source["metadata_sha256"] or metadata["batch_sha256"] != diagnostic_source["batch_sha256"]:
        raise RuntimeError("E-5.4 diagnostic custody mismatch")
    if sha256_file(acceptance_path) != diagnostic_source["acceptance_sha256"] or not acceptance["passed"]:
        raise RuntimeError("E-5.4 diagnostic acceptance mismatch")
    diagnostic_root = resolve(diagnostic_source["output_root"])
    tables = {}
    source_reads = []
    for name, filename in TABLE_FILES.items():
        path = diagnostic_root / filename
        expected_hash = metadata["artifacts"].get(_artifact_hash_key(filename))
        actual_hash = sha256_file(path)
        if expected_hash is None or actual_hash != expected_hash:
            raise RuntimeError(f"E-5.4 diagnostic table custody mismatch: {filename}")
        tables[name] = pd.read_csv(path, dtype={"stock_code": str})
        source_reads.append({"table": name, "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": actual_hash, "read_count": 1})
    matrix, failures, outcome = apply_candidate_gates(gate_config, tables, source_contract_pass=True)

    output_root.mkdir(parents=True, exist_ok=False)
    matrix_path = output_root / "candidate_gate_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    failures_path = output_root / "candidate_gate_failures.json"
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    recommendation = {
        **outcome, "formed_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_config_sha256": application["gate_config_sha256"],
        "gate_freeze_receipt_sha256": application["gate_freeze_receipt_sha256"],
        "three_seed_inference_aggregation": gate_config["three_seed_inference_aggregation"],
        "threshold_relaxation_performed": False, "new_candidate_added": False,
        "new_training_performed": False, "future_or_sealed_data_read": False, "screening_accessed": False,
    }
    recommendation["recommendation_sha256"] = stable_json_sha256(recommendation)
    recommendation_path = output_root / "unique_candidate_recommendation.json"
    recommendation_path.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_out = {
        "stage": "E-6.2 one-shot frozen gate application",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "one_shot_authorization_consumed": True, "candidate_source_table_reads": source_reads,
        "candidate_source_tables_read_once_each": all(row["read_count"] == 1 for row in source_reads),
        "candidate_count": len(gate_config["candidate_model_ids"]), "eligible_count": outcome["eligible_count"],
        "outcome_status": outcome["status"], "unique_candidate": outcome["unique_candidate"],
        "candidate_ranking_performed": False,
        "frozen_tie_break_applied": outcome["status"] == "UNIQUE_CANDIDATE_BY_FROZEN_TIE_BREAK",
        "new_training_performed": False, "threshold_change_performed": False,
        "future_or_sealed_data_read": False, "screening_accessed": False,
        "application_config_sha256": sha256_file(application_config_path),
        "gate_config_sha256": application["gate_config_sha256"],
        "source_diagnostic_batch_sha256": diagnostic_source["batch_sha256"],
        "artifacts": {
            "candidate_gate_matrix_sha256": sha256_file(matrix_path),
            "candidate_gate_failures_sha256": sha256_file(failures_path),
            "unique_candidate_recommendation_sha256": sha256_file(recommendation_path),
        },
    }
    metadata_out["batch_sha256"] = stable_json_sha256(metadata_out)
    (output_root / "metadata.json").write_text(json.dumps(metadata_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata_out, ensure_ascii=False, indent=2))
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(config_path))


if __name__ == "__main__":
    main()
