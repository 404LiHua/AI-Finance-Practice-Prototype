from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import load_config, write_json  # noqa: E402


def prediction_stability(output_dir: Path, variant: str, seeds: list[int]) -> dict[str, float]:
    merged = None
    prediction_columns = []
    for seed in seeds:
        frame = pd.read_csv(output_dir / f"{variant}_seed{seed}" / "predictions.csv")
        column = f"prediction_{seed}"
        prediction_columns.append(column)
        selected = frame[["stock_code", "target_date", "prediction"]].rename(columns={"prediction": column})
        merged = selected if merged is None else merged.merge(
            selected, on=["stock_code", "target_date"], how="inner", validate="one_to_one",
        )
    assert merged is not None and len(merged) == 120
    correlations = [
        float(merged[left].corr(merged[right]))
        for left, right in itertools.combinations(prediction_columns, 2)
    ]
    values = merged[prediction_columns].to_numpy(dtype=float)
    return {
        "mean_pairwise_prediction_correlation": float(np.mean(correlations)),
        "mean_cross_seed_prediction_std": float(values.std(axis=1, ddof=1).mean()),
    }


def main() -> None:
    base = load_config(REPO_ROOT / "stage_c/configs/graph_frequency_v1.json", REPO_ROOT)
    matrix = json.loads((REPO_ROOT / "stage_c/configs/stabilization_v1.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in matrix["seeds"]]
    output_dir = Path(base["output_root"]) / matrix["experiment_name"]
    config_dir = output_dir / "resolved_variant_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for variant, overrides in matrix["variants"].items():
        for seed in seeds:
            config = {
                **base,
                **overrides,
                "experiment_name": matrix["experiment_name"],
                "variant_id": variant,
                "seed": seed,
                "save_plots": False,
                "stabilization_scope": "bounded_stage_c_graph_v1",
            }
            config.pop("description", None)
            path = config_dir / f"{variant}_seed{seed}.json"
            write_json(path, config)
            completed = subprocess.run(
                [sys.executable, str(REPO_ROOT / "stage_c/run_prototype.py"), "--config", str(path)],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Failed {variant} seed={seed}\n{completed.stdout}\n{completed.stderr}")
            print(f"completed {variant} seed={seed}")

    rows: list[dict[str, Any]] = []
    for variant, overrides in matrix["variants"].items():
        for seed in seeds:
            payload = json.loads((output_dir / f"{variant}_seed{seed}" / "metrics.json").read_text(encoding="utf-8"))
            aggregate = payload["metrics"]["validation"]["aggregate"]
            rows.append({
                "variant": variant,
                "description": overrides["description"],
                "seed": seed,
                "mae": aggregate["mae"],
                "rmse": aggregate["rmse"],
                "direction_accuracy": aggregate["direction_accuracy"],
                "direction_f1": aggregate["direction_f1"],
            })
    details = pd.DataFrame(rows).sort_values(["variant", "seed"])
    summary = details.groupby(["variant", "description"], as_index=False).agg(
        runs=("seed", "count"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_f1_mean=("direction_f1", "mean"),
    )
    stability = {
        variant: prediction_stability(output_dir, variant, seeds)
        for variant in matrix["variants"]
    }
    summary["mean_pairwise_prediction_correlation"] = summary["variant"].map(
        lambda variant: stability[variant]["mean_pairwise_prediction_correlation"]
    )
    summary["mean_cross_seed_prediction_std"] = summary["variant"].map(
        lambda variant: stability[variant]["mean_cross_seed_prediction_std"]
    )
    summary = summary.sort_values("mae_mean")

    temporal = summary.loc[summary["variant"] == "temporal_only_control"].iloc[0]
    fixed = summary.loc[summary["variant"] == "fixed_temporal_graph_control"].iloc[0]
    learned = summary[summary["variant"].isin({
        "deterministic_topk2", "deterministic_topk4", "annealed_gumbel_topk2",
    })]
    eligible = learned[
        (learned["mae_mean"] < min(temporal["mae_mean"], fixed["mae_mean"]))
        & (learned["rmse_mean"] < min(temporal["rmse_mean"], fixed["rmse_mean"]))
        & (learned["mae_std"] <= temporal["mae_std"])
    ]
    decision = "INTRODUCE_CROSS_SECTIONAL_NODE_PROTOTYPE" if not eligible.empty else "DEFER_CROSS_SECTIONAL_NODE_DESIGN"
    rationale = (
        "At least one stabilized learned graph passed the frozen accuracy and stability gate."
        if not eligible.empty else
        "No stabilized learned graph beat both controls on MAE and RMSE while matching temporal-control MAE stability."
    )

    details.to_csv(output_dir / "stabilization_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "stabilization_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "stabilization_decision.json", {
        "scope": "30-stock development validation graph stabilization",
        "seeds": seeds,
        "test_status": "reserved_not_evaluated_or_used_for_selection",
        "cross_sectional_gate": matrix["cross_sectional_gate"],
        "decision": decision,
        "rationale": rationale,
        "eligible_variants": eligible["variant"].tolist(),
        "summary": summary.to_dict(orient="records"),
    })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(summary["variant"], summary["mae_mean"], yerr=summary["mae_std"], capsize=4)
    axis.set(ylabel="Validation MAE (mean ± std)", title="Graph stabilization comparison")
    axis.tick_params(axis="x", rotation=22)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "stabilization_mae_comparison.png", dpi=160)
    plt.close(figure)
    print(summary.to_string(index=False))
    print(json.dumps({"decision": decision, "rationale": rationale}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

