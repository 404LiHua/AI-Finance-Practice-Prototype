from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import evaluate_predictions, write_json  # noqa: E402


SEEDS = (20260723, 20260724, 20260725)


def main() -> None:
    source_root = REPO_ROOT / "outputs/experiments/stage_c_30stocks_graph_stabilization"
    output_root = REPO_ROOT / "outputs/experiments/stage_c_30stocks_recommended_v2"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        temporal_dir = source_root / f"temporal_only_control_seed{seed}"
        fixed_dir = source_root / f"fixed_temporal_graph_control_seed{seed}"
        temporal = pd.read_csv(temporal_dir / "predictions.csv")
        fixed = pd.read_csv(fixed_dir / "predictions.csv")
        keys = ["stock_code", "target_date"]
        fixed_prediction = fixed[keys + ["prediction"]].rename(columns={"prediction": "fixed_graph_prediction"})
        combined = temporal.merge(fixed_prediction, on=keys, how="inner", validate="one_to_one")
        if len(combined) != 120:
            raise ValueError(f"seed={seed} ensemble requires 120 aligned validation rows")
        combined = combined.rename(columns={"prediction": "temporal_prediction"})
        combined["prediction"] = 0.5 * combined["temporal_prediction"] + 0.5 * combined["fixed_graph_prediction"]
        combined["predicted_close"] = combined["model_close"] * (1.0 + combined["prediction"])
        combined["absolute_error"] = (combined["prediction"] - combined["target_return"]).abs()
        metrics = evaluate_predictions(combined)
        run_id = f"fixed_control_ensemble_v2_seed{seed}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        write_json(run_dir / "metrics.json", {
            "run_id": run_id,
            "model": "fixed_control_ensemble_v2",
            "seed": seed,
            "split": "validation",
            "metrics": {"validation": metrics},
            "test_status": "reserved_not_evaluated_or_used_for_selection",
            "selection_note": "Development validation-selected structure; not independent evidence.",
        })
        write_json(run_dir / "model_manifest.json", {
            "model": "fixed_control_ensemble_v2",
            "combination": "0.5 * temporal_only_control + 0.5 * fixed_temporal_graph_control",
            "trainable_ensemble_parameters": 0,
            "components": [
                {
                    "component": "temporal_only",
                    "weight": 0.5,
                    "run_dir": temporal_dir.relative_to(REPO_ROOT).as_posix(),
                    "model": (temporal_dir / "model.pt").relative_to(REPO_ROOT).as_posix(),
                },
                {
                    "component": "fixed_temporal_graph",
                    "weight": 0.5,
                    "run_dir": fixed_dir.relative_to(REPO_ROOT).as_posix(),
                    "model": (fixed_dir / "model.pt").relative_to(REPO_ROOT).as_posix(),
                },
            ],
        })
        aggregate = metrics["aggregate"]
        rows.append({"seed": seed, **aggregate})

    details = pd.DataFrame(rows)
    summary_row = {
        "model": "fixed_control_ensemble_v2",
        "runs": len(details),
        "mae_mean": details["mae"].mean(), "mae_std": details["mae"].std(),
        "rmse_mean": details["rmse"].mean(), "rmse_std": details["rmse"].std(),
        "direction_accuracy_mean": details["direction_accuracy"].mean(),
        "direction_accuracy_std": details["direction_accuracy"].std(),
        "direction_f1_mean": details["direction_f1"].mean(),
        "direction_f1_std": details["direction_f1"].std(),
    }
    details.to_csv(output_root / "recommended_v2_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary_row]).to_csv(output_root / "recommended_v2_summary.csv", index=False, encoding="utf-8-sig")

    baseline = pd.read_csv(REPO_ROOT / "outputs/experiments/stage_c_graph_frequency_v1/unified_baseline_summary.csv")
    stabilization = pd.read_csv(source_root / "stabilization_summary.csv")
    comparison = pd.concat([
        baseline.rename(columns={"model": "model"})[
            ["model", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "direction_accuracy_mean", "direction_f1_mean"]
        ],
        stabilization[stabilization["variant"].isin(["temporal_only_control", "fixed_temporal_graph_control"])].rename(columns={"variant": "model"})[
            ["model", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "direction_accuracy_mean", "direction_f1_mean"]
        ],
        pd.DataFrame([summary_row])[
            ["model", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "direction_accuracy_mean", "direction_f1_mean"]
        ],
    ], ignore_index=True).sort_values("mae_mean")
    comparison.to_csv(output_root / "recommended_v2_unified_comparison.csv", index=False, encoding="utf-8-sig")
    v1 = comparison[comparison["model"] == "graph_frequency_v1"].iloc[0]
    v2 = comparison[comparison["model"] == "fixed_control_ensemble_v2"].iloc[0]
    write_json(output_root / "recommended_v2_decision.json", {
        "decision": "PROMOTE_FIXED_CONTROL_ENSEMBLE_V2_OVER_GRAPH_FREQUENCY_V1",
        "criterion": "fixed ensemble improves both three-seed mean MAE and RMSE versus v1 without learned ensemble parameters",
        "mae_improvement_vs_v1_pct": float((v1["mae_mean"] - v2["mae_mean"]) / v1["mae_mean"] * 100),
        "rmse_improvement_vs_v1_pct": float((v1["rmse_mean"] - v2["rmse_mean"]) / v1["rmse_mean"] * 100),
        "recommended_model": "fixed_control_ensemble_v2",
        "cross_sectional_node_design": "deferred",
    })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = comparison[comparison["model"].isin([
        "frets_return_l4", "minimalist_price_only_l8", "naive", "graph_frequency_v1",
        "temporal_only_control", "fixed_temporal_graph_control", "fixed_control_ensemble_v2",
    ])]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(selected["model"], selected["mae_mean"], yerr=selected["mae_std"], capsize=4)
    axis.set(ylabel="Validation MAE (mean ± std)", title="Recommended Stage C v2 comparison")
    axis.tick_params(axis="x", rotation=24)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_root / "recommended_v2_comparison.png", dpi=160)
    plt.close(figure)
    print(comparison.to_string(index=False))
    print(json.dumps(json.loads((output_root / "recommended_v2_decision.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
