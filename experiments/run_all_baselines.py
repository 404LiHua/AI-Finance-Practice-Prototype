from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import load_config, write_json  # noqa: E402
from experiments.runner import run_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all configured Stage B baselines and seeds.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments/configs/stage_b_baselines.json")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config.resolve(), REPO_ROOT)
    models = args.models or config["models"]
    seeds = args.seeds or config["seeds"]
    results = [run_model(config, model, int(seed), REPO_ROOT) for model in models for seed in seeds]
    rows = []
    for result in results:
        for split in ("validation", "test"):
            rows.append({
                "run_id": result["run_id"], "model": result["model"], "seed": result["seed"],
                "split": split, "duration_seconds": result["duration_seconds"],
                **result["metrics"][split]["aggregate"],
            })
    output_dir = Path(config["output_root"]) / config["experiment_name"]
    table = pd.DataFrame(rows).sort_values(["split", "mae", "model", "seed"])
    table.to_csv(output_dir / "baseline_results.csv", index=False, encoding="utf-8-sig")
    summary = table.groupby(["model", "split"], as_index=False).agg(
        runs=("seed", "count"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_f1_mean=("direction_f1", "mean"), duration_seconds_mean=("duration_seconds", "mean"),
    )
    summary.to_csv(output_dir / "baseline_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "completed_runs.json", results)
    print(json.dumps({"runs": len(results), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
