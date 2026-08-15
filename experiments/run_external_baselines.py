from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import (  # noqa: E402
    DataBundle, Timer, create_logger, environment_info, evaluate_predictions,
    load_config, prediction_frame, set_global_seed, write_json,
)
from experiments.external_adapters import FreTSAdapter, TimeGNNAdapter  # noqa: E402


def build_adapter(name: str, config: dict[str, Any], seed: int) -> Any:
    if name == "frets":
        return FreTSAdapter(config["frets"], config["upstream"]["frets"], seed)
    if name == "timegnn":
        return TimeGNNAdapter(config["timegnn"], config["upstream"]["timegnn"], seed)
    raise ValueError(f"Unsupported external adapter: {name}")


def run_adapter(config: dict[str, Any], model_name: str, seed: int) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{model_name}_seed{seed}"
    output_dir = Path(config["output_root"]) / config["experiment_name"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir)
    resolved = dict(config)
    resolved.update({"selected_model": model_name, "selected_seed": seed, "run_id": run_id})
    write_json(output_dir / "resolved_config.json", resolved)
    write_json(output_dir / "environment.json", environment_info(REPO_ROOT))
    write_json(output_dir / "seeds.json", set_global_seed(seed))
    logger.info("Starting external adapter model=%s seed=%d", model_name, seed)
    with Timer() as timer:
        data = DataBundle.load(Path(config["data_root"]))
        logger.info(
            "Loaded identical samples train=%d validation=%d test=%d",
            len(data.samples["train"]), len(data.samples["validation"]), len(data.samples["test"]),
        )
        adapter = build_adapter(model_name, config, seed)
        adapter.fit(data, output_dir, logger)
        predictions = []
        metrics: dict[str, Any] = {}
        for split in ("validation", "test"):
            values = adapter.predict(data, split)
            frame = prediction_frame(data.samples[split], values, split)
            predictions.append(frame)
            metrics[split] = evaluate_predictions(frame)
            logger.info("%s metrics: %s", split, metrics[split]["aggregate"])
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False, encoding="utf-8-sig",
    )
    result = {
        "run_id": run_id,
        "model": model_name,
        "seed": seed,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": timer.seconds,
        "sample_counts": {split: len(frame) for split, frame in data.samples.items()},
        "adapter_feature_columns": adapter.feature_columns,
        "metrics": metrics,
        "output_dir": str(output_dir),
        "evidence_class": "SELECTION_EXPOSED_NON_INDEPENDENT",
    }
    write_json(output_dir / "metrics.json", result)
    logger.info("Completed external adapter in %.3f seconds", timer.seconds)
    return result


def aggregate_all_runs(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_runs = []
    for metrics_path in sorted(experiment_dir.glob("*_seed*/metrics.json")):
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not {"model", "seed", "metrics"} <= result.keys():
            continue
        all_runs.append(result)
        for split in ("validation", "test"):
            if split not in result["metrics"]:
                continue
            rows.append({
                "run_id": result["run_id"], "model": result["model"], "seed": result["seed"],
                "split": split, "duration_seconds": result["duration_seconds"],
                **result["metrics"][split]["aggregate"],
            })
    table = pd.DataFrame(rows).sort_values(["split", "mae", "model", "seed"])
    summary = table.groupby(["model", "split"], as_index=False).agg(
        runs=("seed", "count"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_f1_mean=("direction_f1", "mean"), duration_seconds_mean=("duration_seconds", "mean"),
    )
    table.to_csv(experiment_dir / "baseline_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(experiment_dir / "baseline_summary.csv", index=False, encoding="utf-8-sig")
    write_json(experiment_dir / "all_completed_runs.json", all_runs)
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FreTS/Time-GNN on the unified Stage B splits.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments/configs/external_adapters.json")
    parser.add_argument("--models", nargs="+", choices=["frets", "timegnn"], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config.resolve(), REPO_ROOT)
    models = args.models or config["models"]
    seeds = args.seeds or config["seeds"]
    results = [run_adapter(config, model, int(seed)) for model in models for seed in seeds]
    experiment_dir = Path(config["output_root"]) / config["experiment_name"]
    _, summary = aggregate_all_runs(experiment_dir)
    print(json.dumps({"new_runs": len(results), "output_dir": str(experiment_dir)}, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
