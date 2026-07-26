"""Accept sealed E-6.2 outputs without rereading source candidate diagnostics."""

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

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.run_e3_training_checks import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "outputs/stage_e/e6_candidate_gate_application_acceptance_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate_config = json.loads(resolve(config["gate_config"]).read_text(encoding="utf-8"))
    root = resolve(config["output_root"])
    matrix_path = root / "candidate_gate_matrix.csv"
    failures_path = root / "candidate_gate_failures.json"
    recommendation_path = root / "unique_candidate_recommendation.json"
    metadata_path = root / "metadata.json"
    matrix = pd.read_csv(matrix_path)
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared_batch = metadata.pop("batch_sha256")
    eligible = matrix.loc[matrix["eligible"].astype(bool), "model_id"].astype(str).tolist()
    failed_counts_match = all(
        int(row.failed_gate_count) == len(failures[str(row.model_id)])
        and bool(row.eligible) == (len(failures[str(row.model_id)]) == 0)
        for row in matrix.itertuples(index=False)
    )
    source_reads = metadata["candidate_source_table_reads"]
    checks = {
        "application_config_hash_valid": metadata["application_config_sha256"] == sha256_file(config_path),
        "gate_config_hash_valid": sha256_file(resolve(config["gate_config"])) == config["gate_config_sha256"],
        "freeze_receipt_hash_valid": sha256_file(resolve(config["gate_freeze_receipt"])) == config["gate_freeze_receipt_sha256"],
        "implementation_hashes_valid": (
            sha256_file(resolve(config["gate_implementation"])) == config["gate_implementation_sha256"]
            and sha256_file(resolve(config["one_shot_executor"])) == config["one_shot_executor_sha256"]
        ),
        "exactly_eight_frozen_candidates": sorted(matrix["model_id"].astype(str)) == sorted(gate_config["candidate_model_ids"]),
        "all_hard_gate_columns_present": all(column.startswith("gate_") for column in matrix.columns if column.startswith("gate_")) and len([column for column in matrix.columns if column.startswith("gate_")]) == int(recommendation["gate_column_count"]),
        "failure_lists_match_matrix": set(failures) == set(gate_config["candidate_model_ids"]) and failed_counts_match,
        "eligible_set_matches_recommendation": sorted(eligible) == sorted(recommendation["eligible_models"]),
        "formal_outcome_is_valid": (
            (len(eligible) == 0 and recommendation["status"] == "FORMAL_NO_PROMOTABLE_CANDIDATE" and recommendation["unique_candidate"] is None)
            or (len(eligible) >= 1 and recommendation["status"] in {"UNIQUE_CANDIDATE_RECOMMENDATION", "UNIQUE_CANDIDATE_BY_FROZEN_TIE_BREAK"} and recommendation["unique_candidate"] in eligible)
        ),
        "source_tables_read_once_each": len(source_reads) == 10 and all(int(row["read_count"]) == 1 for row in source_reads),
        "sealed_artifact_hashes_valid": (
            sha256_file(matrix_path) == metadata["artifacts"]["candidate_gate_matrix_sha256"]
            and sha256_file(failures_path) == metadata["artifacts"]["candidate_gate_failures_sha256"]
            and sha256_file(recommendation_path) == metadata["artifacts"]["unique_candidate_recommendation_sha256"]
        ),
        "metadata_batch_hash_valid": declared_batch == stable_json_sha256(metadata),
        "no_ranking_training_threshold_change_future_or_screening": (
            not metadata["candidate_ranking_performed"] and not metadata["new_training_performed"]
            and not metadata["threshold_change_performed"] and not metadata["future_or_sealed_data_read"]
            and not metadata["screening_accessed"] and not recommendation["threshold_relaxation_performed"]
            and not recommendation["new_candidate_added"]
        ),
    }
    report = {
        "stage": "E-6.2 one-shot frozen gate application acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "outcome_status": recommendation["status"],
        "unique_candidate": recommendation["unique_candidate"],
        "eligible_models": eligible,
        "config_sha256": sha256_file(config_path),
        "sealed_metadata_sha256": sha256_file(metadata_path),
        "source_candidate_diagnostics_reread": False,
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
