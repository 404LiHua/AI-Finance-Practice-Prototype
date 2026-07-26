"""Unified E-4 protocol closure acceptance without overriding failed promotion gates."""

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


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e4_closure_acceptance_100stocks_v1.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load(config_path)
    control_root = resolve(config["paths"]["output_root"])
    control_path = control_root / "results.json"
    control = load(control_path)
    fixed_acceptance = load(REPO_ROOT / "outputs/stage_e/e4_fixed_graph_stabilization_acceptance_100stocks_v1.json")
    architecture = load(REPO_ROOT / "outputs/stage_e/e4_architecture_acceptance_v1.json")
    adapter = load(REPO_ROOT / "outputs/stage_e/e4_adapter_acceptance_100stocks_v1.json")
    ablation = load(REPO_ROOT / "outputs/stage_e/e4_ablation_acceptance_100stocks_v1.json")
    fold = pd.read_csv(control_root / "fold_results.csv")
    payload = dict(control)
    declared_batch = payload.pop("batch_sha256", "")
    artifact_names = {
        "fold_results_sha256": "fold_results.csv",
        "predictions_sha256": "predictions.csv.gz",
        "diagnostics_per_stock_sha256": "diagnostics_per_stock.csv",
        "diagnostics_industry_sha256": "diagnostics_industry.csv",
        "diagnostics_market_cap_sha256": "diagnostics_market_cap.csv",
        "diagnostics_return_decile_sha256": "diagnostics_return_decile.csv",
        "diagnostics_control_disagreement_sha256": "diagnostics_control_disagreement.csv",
    }
    protocol_checks = {
        "e3_still_accepted": load(REPO_ROOT / "outputs/stage_e/e3_acceptance_v1.json")["passed"],
        "architecture_accepted": architecture["passed"],
        "adapter_accepted": adapter["passed"],
        "first_layer_ablation_accepted": ablation["passed"],
        "fixed_graph_process_accepted": fixed_acceptance["passed"],
        "fixed_graph_stability_failure_preserved": not fixed_acceptance["stability_pass"],
        "two_controls_three_seeds_three_folds": len(fold) == 18 and set(fold["variant"].astype(str)) == set(control["controls"]),
        "control_batch_hash_valid": declared_batch == stable_json_sha256(payload),
        "diagnostic_hashes_valid": all(sha256_file(control_root / name) == control["artifacts"][key] for key, name in artifact_names.items()),
        "future_or_sealed_data_not_read": not control.get("future_or_sealed_data_read", True),
        "three_hundred_expansion_remained_disabled": not control.get("allow_300_stock_graph_frequency_text_expansion", True),
    }
    completion_checks = {
        "at_least_one_stable_control": control["at_least_one_stable_control"],
        "graph_frequency_text_three_seed_pass": fixed_acceptance["stability_pass"],
        "three_hundred_stock_graph_frequency_text_completed": False,
        "grouped_diagnostics_completed": protocol_checks["diagnostic_hashes_valid"],
    }
    report = {
        "stage": "E-4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_pass": all(protocol_checks.values()),
        "stage_completed": all(completion_checks.values()),
        "status": "COMPLETED" if all(completion_checks.values()) else "NOT_COMPLETED_STABILITY_GATE_FAILURE",
        "protocol_checks": protocol_checks,
        "completion_checks": completion_checks,
        "stable_controls": control["stable_controls"],
        "allow_300_stock_expansion": False,
        "control_results_sha256": sha256_file(control_path),
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["protocol_pass"] else 2)


if __name__ == "__main__":
    main()
