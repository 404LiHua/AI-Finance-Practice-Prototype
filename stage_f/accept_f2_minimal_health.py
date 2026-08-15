"""Independently accept the authorized F-2.1 minimal GAN training health receipt."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stage_e.hashing import sha256_file, stable_json_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt_path = resolve(config["paths"]["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_hashes_valid = all(
        sha256_file(resolve(item["path"])) == item["sha256"]
        for item in config["source_registry"].values()
    )
    checks = {
        "source_hashes_valid": source_hashes_valid,
        "receipt_pass": receipt["status"] == "PASS",
        "all_health_checks_pass": receipt["passed_checks"] == receipt["required_checks"] == 4,
        "all_protocol_checks_pass": receipt["passed_protocol_checks"] == receipt["required_protocol_checks"],
        "config_hash_matches": receipt["config_sha256"] == sha256_file(config_path),
        "unique_candidate_unchanged": receipt["candidate_id"] == "stock_node_gwnet_bounded_cwgan_gp_l8",
        "optimizer_steps_were_minimal_only": receipt["optimizer_steps_executed"]
        and receipt["synthetic_minimal_training_executed"]
        and not receipt["formal_fold_training_executed"],
        "no_additional_seed_training": not receipt["additional_seed_training_executed"],
        "no_ranking_or_metric_read": not receipt["candidate_metrics_read"]
        and not receipt["candidate_ranking_performed"],
        "future_data_not_accessed": not receipt["screening_accessed"] and not receipt["final_accessed"],
    }
    acceptance = {
        "stage": "F-2.1 minimal GAN training health independent acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "required_checks": len(checks),
        "config_sha256": sha256_file(config_path),
        "receipt_sha256": sha256_file(receipt_path),
        "formal_fold_training_authorized": False,
        "screening_authorized": False,
        "final_authorized": False,
        "next_action": "request separate authorization before F-2.2 single-seed three-fold engineering training",
    }
    acceptance["acceptance_sha256"] = stable_json_sha256(acceptance)
    output = resolve(config["paths"]["acceptance"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if acceptance["status"] != "PASS":
        raise RuntimeError("F-2.1 independent acceptance failed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(config_path))


if __name__ == "__main__":
    main()
