from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import load_config, write_json  # noqa: E402


def main() -> None:
    base_path = REPO_ROOT / "stage_c/configs/graph_frequency_v1.json"
    matrix_path = REPO_ROOT / "stage_c/configs/ablations_v1.json"
    base = load_config(base_path, REPO_ROOT)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    experiment_name = str(matrix["experiment_name"])
    output_dir = Path(base["output_root"]) / experiment_name
    config_dir = output_dir / "resolved_variant_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for variant_id, overrides in matrix["variants"].items():
        for seed in matrix["seeds"]:
            config = {
                **base,
                **overrides,
                "experiment_name": experiment_name,
                "variant_id": variant_id,
                "seed": int(seed),
                "save_plots": False,
                "ablation_scope": "bounded_stage_c_v1",
            }
            config.pop("description", None)
            config_path = config_dir / f"{variant_id}_seed{seed}.json"
            write_json(config_path, config)
            completed = subprocess.run(
                [sys.executable, str(REPO_ROOT / "stage_c/run_prototype.py"), "--config", str(config_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Ablation failed: {variant_id} seed={seed}\n{completed.stdout}\n{completed.stderr}"
                )
            print(f"completed {variant_id} seed={seed}")

    rows: list[dict[str, Any]] = []
    for variant_id, overrides in matrix["variants"].items():
        for seed in matrix["seeds"]:
            metrics_path = output_dir / f"{variant_id}_seed{seed}" / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            aggregate = payload["metrics"]["validation"]["aggregate"]
            rows.append({
                "variant": variant_id,
                "description": overrides["description"],
                "seed": seed,
                "mae": aggregate["mae"],
                "rmse": aggregate["rmse"],
                "direction_accuracy": aggregate["direction_accuracy"],
                "direction_f1": aggregate["direction_f1"],
                "duration_seconds": payload["duration_seconds"],
            })
    details = pd.DataFrame(rows).sort_values(["variant", "seed"])
    summary = details.groupby(["variant", "description"], as_index=False).agg(
        runs=("seed", "count"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_accuracy_std=("direction_accuracy", "std"),
        direction_f1_mean=("direction_f1", "mean"),
        direction_f1_std=("direction_f1", "std"),
    )
    full_mae = float(summary.loc[summary["variant"] == "full_graph_frequency_k2", "mae_mean"].iloc[0])
    full_rmse = float(summary.loc[summary["variant"] == "full_graph_frequency_k2", "rmse_mean"].iloc[0])
    summary["mae_delta_vs_full_pct"] = (summary["mae_mean"] - full_mae) / full_mae * 100.0
    summary["rmse_delta_vs_full_pct"] = (summary["rmse_mean"] - full_rmse) / full_rmse * 100.0
    summary = summary.sort_values("mae_mean")
    details.to_csv(output_dir / "ablation_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "ablation_summary.json", {
        "scope": "30-stock development validation bounded ablation",
        "seeds": matrix["seeds"],
        "test_status": "reserved_not_evaluated_or_used_for_selection",
        "summary": summary.to_dict(orient="records"),
    })

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(summary["variant"], summary["mae_mean"], yerr=summary["mae_std"], capsize=4)
    axis.set(ylabel="Validation MAE (mean ± std)", title="Stage C bounded structural ablations")
    axis.tick_params(axis="x", rotation=24)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "ablation_mae_comparison.png", dpi=160)
    plt.close(figure)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
