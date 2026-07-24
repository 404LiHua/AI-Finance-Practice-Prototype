from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260723, 20260724, 20260725)


def main() -> None:
    config_path = REPO_ROOT / "stage_c/configs/improved_v2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = REPO_ROOT / config["output_root"] / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "stage_c/run_prototype.py"), "--config", str(config_path), "--seed", str(seed)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"v2 failed seed={seed}\n{completed.stdout}\n{completed.stderr}")
        print(f"completed fixed_temporal_timegraph_v2 seed={seed}")

    rows = []
    for seed in SEEDS:
        payload = json.loads((output_dir / f"fixed_temporal_timegraph_v2_seed{seed}" / "metrics.json").read_text(encoding="utf-8"))
        aggregate = payload["metrics"]["validation"]["aggregate"]
        rows.append({"seed": seed, **aggregate})
    details = pd.DataFrame(rows)
    summary = details.agg({
        "mae": ["mean", "std"],
        "rmse": ["mean", "std"],
        "direction_accuracy": ["mean", "std"],
        "direction_f1": ["mean", "std"],
    })
    details.to_csv(output_dir / "v2_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "v2_summary.csv", encoding="utf-8-sig")

    comparison_sources = {
        "graph_frequency_v1": REPO_ROOT / "outputs/experiments/stage_c_graph_frequency_v1/unified_baseline_summary.csv",
        "stabilization": REPO_ROOT / "outputs/experiments/stage_c_30stocks_graph_stabilization/stabilization_summary.csv",
    }
    baseline = pd.read_csv(comparison_sources["graph_frequency_v1"])
    stabilization = pd.read_csv(comparison_sources["stabilization"])
    comparison = pd.concat([
        baseline[baseline["model"].isin(["naive", "frets_return_l4", "minimalist_price_only_l8", "graph_frequency_v1"])].rename(columns={"model": "model"})[
            ["model", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "direction_accuracy_mean", "direction_f1_mean"]
        ],
        stabilization[stabilization["variant"].isin(["temporal_only_control", "fixed_temporal_graph_control"])].rename(columns={"variant": "model"})[
            ["model", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "direction_accuracy_mean", "direction_f1_mean"]
        ],
        pd.DataFrame([{
            "model": "fixed_temporal_timegraph_v2",
            "mae_mean": details["mae"].mean(), "mae_std": details["mae"].std(),
            "rmse_mean": details["rmse"].mean(), "rmse_std": details["rmse"].std(),
            "direction_accuracy_mean": details["direction_accuracy"].mean(),
            "direction_f1_mean": details["direction_f1"].mean(),
        }]),
    ], ignore_index=True).sort_values("mae_mean")
    comparison.to_csv(output_dir / "v2_unified_comparison.csv", index=False, encoding="utf-8-sig")
    v1 = comparison[comparison["model"] == "graph_frequency_v1"].iloc[0]
    v2 = comparison[comparison["model"] == "fixed_temporal_timegraph_v2"].iloc[0]
    promoted = bool(v2["mae_mean"] < v1["mae_mean"] and v2["rmse_mean"] < v1["rmse_mean"])
    decision = {
        "decision": "PROMOTE_V2_OVER_V1" if promoted else "RETAIN_V1_AND_CONTROLS",
        "criterion": "v2 must improve both three-seed mean MAE and RMSE versus graph-frequency v1",
        "v2_mae_improvement_pct": float((v1["mae_mean"] - v2["mae_mean"]) / v1["mae_mean"] * 100),
        "v2_rmse_improvement_pct": float((v1["rmse_mean"] - v2["rmse_mean"]) / v1["rmse_mean"] * 100),
        "recommended_model": "fixed_temporal_timegraph_v2" if promoted else "temporal_only_control",
    }
    (output_dir / "v2_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

