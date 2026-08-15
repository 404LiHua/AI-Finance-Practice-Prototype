from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import write_json  # noqa: E402
from stage_d.d3_diagnostics import (  # noqa: E402
    component_disagreement_diagnostics,
    load_diagnostic_config,
    load_prediction_grid,
    per_stock_diagnostics,
    return_group_diagnostics,
    seed_stability_diagnostics,
    select_unique_candidate,
    sha256_file,
)


def run(config_path: Path) -> dict:
    config = load_diagnostic_config(config_path.resolve(), REPO_ROOT)
    source_root = Path(config["source_root"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    source_hashes = json.loads((source_root / "sha256_manifest.json").read_text(encoding="utf-8"))
    for name, expected in source_hashes.items():
        observed = sha256_file(source_root / name)
        if observed != expected:
            raise RuntimeError(f"D-2 source output hash mismatch: {name}")

    metrics = pd.read_csv(source_root / "metrics_by_fold_seed.csv")
    cross_fold = pd.read_csv(source_root / "cross_fold_model_summary.csv")
    predictions, registered = load_prediction_grid(config, REPO_ROOT)
    if set(metrics["model"]) != set(registered) or set(cross_fold["model"]) != set(registered):
        raise RuntimeError("D-3 source candidate set differs from locked D-2 registration")

    gate_table, selection = select_unique_candidate(cross_fold, config)
    stock = per_stock_diagnostics(predictions)
    return_groups = return_group_diagnostics(predictions, config["return_groups"])
    seed_by_model, seed_summary = seed_stability_diagnostics(metrics, predictions)
    disagreement, disagreement_by_fold = component_disagreement_diagnostics(
        predictions, config["component_models"]
    )

    per_fold = pd.read_csv(source_root / "per_fold_summary.csv")
    worst_fold = per_fold.sort_values(
        ["model", "mae_gap_vs_baseline_pct", "mae_mean"],
        ascending=[True, False, False], kind="mergesort",
    ).groupby("model", as_index=False).head(1).sort_values("model")
    worst_fold = worst_fold.rename(columns={"fold_id": "worst_fold_id"})

    winner = selection["recommendation"]["recommended_model"]
    winner_stock = stock[stock["model"] == winner].copy()
    winner_groups = return_groups[return_groups["model"] == winner].copy()
    winner_seed = seed_summary[seed_summary["model"] == winner].iloc[0]
    worst_stock = winner_stock.sort_values("mae", ascending=False).iloc[0]
    tail_groups = winner_groups[winner_groups["return_group"].isin(["negative_tail", "positive_tail"])]
    summary = {
        **selection["recommendation"],
        "source_registered_model_count": len(registered),
        "source_metric_rows": len(metrics),
        "source_prediction_rows": len(predictions),
        "worst_stock": {
            "stock_code": str(worst_stock["stock_code"]),
            "mae": float(worst_stock["mae"]),
            "absolute_error_share_pct": float(worst_stock["absolute_error_share_pct"]),
            "mae_improvement_vs_naive_pct": float(worst_stock["mae_improvement_vs_naive_pct"]),
        },
        "tail_return_groups": tail_groups[[
            "return_group", "samples", "mae", "mae_improvement_vs_naive_pct",
        ]].to_dict(orient="records"),
        "seed_stability": {
            "seed_mae_range": float(winner_seed["seed_mae_range"]),
            "seed_mae_cv": float(winner_seed["seed_mae_cv"]),
            "mean_cross_seed_prediction_std": float(winner_seed["mean_cross_seed_prediction_std"]),
        },
        "candidate_additions": 0,
        "shrinkage_changes": 0,
        "fold_definition_mutated": False,
        "c4_rows_read": 0,
        "future_d_screening_rows_read": 0,
        "warning": "Unique recommendation is rolling-development evidence only; D-4 freeze is still required.",
    }

    outputs = {
        "robust_gate_all_candidates.csv": gate_table,
        "eligible_candidate_ranking.csv": selection["eligible"],
        "per_stock_diagnostics.csv": stock,
        "return_group_diagnostics.csv": return_groups,
        "worst_fold_diagnostics.csv": worst_fold,
        "seed_stability_by_model_seed.csv": seed_by_model,
        "seed_stability_summary.csv": seed_summary,
        "component_disagreement_summary.csv": disagreement,
        "component_disagreement_by_fold.csv": disagreement_by_fold,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_root / name, index=False, encoding="utf-8-sig")
    write_json(output_root / "locked_diagnostic_config.json", config)
    write_json(output_root / "unique_candidate_recommendation.json", summary)
    write_json(output_root / "source_hash_verification.json", source_hashes)
    manifest = {
        "diagnostic_id": config["diagnostic_id"],
        "source_experiment_id": config["source_experiment_id"],
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "registered_models": registered,
        "registered_model_count": len(registered),
        "candidate_additions": 0,
        "shrinkage_changes": 0,
        "c4_rows_read": 0,
        "future_d_screening_rows_read": 0,
        "independent_screening": False,
        "recommended_model": winner,
    }
    write_json(output_root / "evidence_manifest.json", manifest)
    hash_names = [*outputs, "locked_diagnostic_config.json", "unique_candidate_recommendation.json",
                  "source_hash_verification.json", "evidence_manifest.json"]
    write_json(output_root / "sha256_manifest.json", {
        name: sha256_file(output_root / name) for name in hash_names
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked Stage D-3 robustness diagnostics.")
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "stage_d/configs/d3_diagnostics.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
