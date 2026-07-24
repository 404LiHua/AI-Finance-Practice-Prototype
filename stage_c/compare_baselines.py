from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import write_json  # noqa: E402


SEEDS = (20260723, 20260724, 20260725)
MODEL_SOURCES = {
    "naive": (
        "stage_b_30stocks_baselines",
        "naive_seed{seed}",
    ),
    "frets_return_l4": (
        "stage_b_30stocks_bounded_ablations",
        "frets_return_l4_seed{seed}",
    ),
    "minimalist_price_only_l8": (
        "stage_b_30stocks_bounded_ablations",
        "minimalist_price_only_l8_seed{seed}",
    ),
    "graph_frequency_v1": (
        "stage_c_graph_frequency_v1",
        "graph_frequency_v1_seed{seed}",
    ),
}


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "validation" not in payload.get("metrics", {}):
        raise KeyError(f"Result does not contain validation metrics: {path}")
    return payload


def main() -> None:
    experiments_root = REPO_ROOT / "outputs/experiments"
    output_dir = experiments_root / "stage_c_graph_frequency_v1"
    rows: list[dict[str, Any]] = []
    for model, (experiment, run_pattern) in MODEL_SOURCES.items():
        for seed in SEEDS:
            run_id = run_pattern.format(seed=seed)
            metrics_path = experiments_root / experiment / run_id / "metrics.json"
            payload = load_metrics(metrics_path)
            aggregate = payload["metrics"]["validation"]["aggregate"]
            rows.append({
                "model": model,
                "seed": seed,
                "split": "validation",
                "samples": aggregate["samples"],
                "mae": aggregate["mae"],
                "rmse": aggregate["rmse"],
                "mse": aggregate["mse"],
                "direction_accuracy": aggregate["direction_accuracy"],
                "direction_f1": aggregate["direction_f1"],
                "source_metrics": str(metrics_path.resolve()),
            })
    details = pd.DataFrame(rows).sort_values(["model", "seed"])
    if set(details["samples"]) != {120}:
        raise ValueError("Comparison requires identical 120-row validation samples")
    summary = details.groupby("model", as_index=False).agg(
        runs=("seed", "count"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_accuracy_std=("direction_accuracy", "std"),
        direction_f1_mean=("direction_f1", "mean"),
        direction_f1_std=("direction_f1", "std"),
    ).sort_values("mae_mean")
    details.to_csv(output_dir / "unified_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "unified_baseline_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "unified_baseline_comparison.json", {
        "scope": "30-stock development validation comparison",
        "split": "validation",
        "seeds": list(SEEDS),
        "models": list(MODEL_SOURCES),
        "sample_count_per_run": 120,
        "independence_note": "Development comparison only; not independent screening or final evidence.",
        "summary": summary.to_dict(orient="records"),
    })

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(
            summary["model"], summary["mae_mean"],
            yerr=summary["mae_std"].fillna(0.0), capsize=4,
        )
        axis.set(ylabel="Validation MAE (mean ± std)", title="Stage C v1 unified baseline comparison")
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "unified_baseline_comparison.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

