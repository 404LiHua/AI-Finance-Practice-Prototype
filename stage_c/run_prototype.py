from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.bounded_ablations import minimalist_feature_view  # noqa: E402
from experiments.core import (  # noqa: E402
    DataBundle, Timer, create_logger, environment_info, evaluate_predictions,
    load_config, prediction_frame, set_global_seed, write_json,
)
from stage_c.trainer import GraphFrequencyExperiment  # noqa: E402


def save_plots(
    output_dir: Path,
    predictions: pd.DataFrame,
    mean_adjacency: np.ndarray,
    logger: object,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping Stage C plots")
        return

    history = json.loads((output_dir / "training_history.json").read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [row["train_loss"] for row in history], label="train")
    axis.plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axis.set(xlabel="Epoch", ylabel="Huber loss", title="Stage C v1 training curve")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curve.png", dpi=160)
    plt.close(figure)

    ordered = predictions.sort_values(["target_date", "stock_code"]).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(ordered.index, ordered["target_return"], label="actual", linewidth=1.2)
    axis.plot(ordered.index, ordered["prediction"], label="prediction", linewidth=1.2)
    axis.set(xlabel="Validation samples ordered by date/stock", ylabel="Next-week return", title="Prediction comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(mean_adjacency, cmap="viridis", aspect="auto")
    axis.set(xlabel="Destination week node", ylabel="Source week node", title="Mean learned adjacency")
    figure.colorbar(image, ax=axis, label="Mean edge weight")
    figure.tight_layout()
    figure.savefig(output_dir / "mean_adjacency_heatmap.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage C graph-frequency v1 prototype.")
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "stage_c/configs/graph_frequency_v1.json",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the configured frozen seed.")
    args = parser.parse_args()
    config = load_config(args.config.resolve(), REPO_ROOT)
    seed = int(args.seed if args.seed is not None else config["seed"])
    config["seed"] = seed
    variant_id = str(config.get("variant_id", "graph_frequency_v1"))
    run_id = f"{variant_id}_seed{seed}"
    output_dir = Path(config["output_root"]) / config["experiment_name"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir)
    write_json(output_dir / "resolved_config.json", {**config, "run_id": run_id})
    write_json(output_dir / "environment.json", environment_info(REPO_ROOT))
    write_json(output_dir / "seeds.json", set_global_seed(seed))

    started_at = datetime.now(timezone.utc).isoformat()
    with Timer() as timer:
        data = DataBundle.load(Path(config["data_root"]))
        stock_count = int(data.panel["stock_code"].nunique())
        if stock_count != 30:
            raise ValueError(f"Stage C v1 is frozen to 30 stocks, found {stock_count}")
        data = minimalist_feature_view(data, str(config["feature_set"]))
        logger.info(
            "Loaded 30-stock data train=%d validation=%d test_reserved=%d features=%d",
            len(data.samples["train"]), len(data.samples["validation"]),
            len(data.samples["test"]), len(data.feature_columns),
        )
        model = GraphFrequencyExperiment(config, seed)
        model.fit(data, output_dir, logger)
        prediction_frames = []
        metrics = {}
        adjacency_rows = []
        gate_rows = []
        for split in config.get("evaluation_splits", ["validation"]):
            prediction, adjacency, gate = model.predict_with_diagnostics(data, split)
            frame = prediction_frame(data.samples[split], prediction, split)
            prediction_frames.append(frame)
            metrics[split] = evaluate_predictions(frame)
            adjacency_rows.append(adjacency)
            gate_rows.append(gate)
            logger.info("%s metrics: %s", split, metrics[split]["aggregate"])

        all_adjacency = np.concatenate(adjacency_rows)
        mean_adjacency = all_adjacency.mean(axis=0)
        np.save(output_dir / "mean_adjacency.npy", mean_adjacency)
        pd.DataFrame(mean_adjacency).to_csv(
            output_dir / "mean_adjacency.csv", index=False, encoding="utf-8-sig",
        )
        gate_values = np.concatenate(gate_rows)
        diagnostics = {
            "mean_gate": float(gate_values.mean()),
            "mean_sample_adjacency_density": float((all_adjacency > 1e-8).mean()),
            "mean_self_loop_weight": float(np.diag(mean_adjacency).mean()),
            "evaluation_splits": list(config.get("evaluation_splits", ["validation"])),
            "reserved_splits": [split for split in ("validation", "test") if split not in config.get("evaluation_splits", [])],
        }
        write_json(output_dir / "graph_diagnostics.json", diagnostics)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        output_dir / "predictions.csv", index=False, encoding="utf-8-sig",
    )
    if bool(config.get("save_plots", True)):
        save_plots(output_dir, predictions, mean_adjacency, logger)
    result = {
        "run_id": run_id,
        "stage": "C-v1-prototype",
        "variant_id": variant_id,
        "seed": seed,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": timer.seconds,
        "sample_counts": {split: len(frame) for split, frame in data.samples.items()},
        "feature_columns": data.feature_columns,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "output_dir": str(output_dir),
        "conclusion_scope": "engineering smoke result only; not a final model conclusion",
    }
    write_json(output_dir / "metrics.json", result)
    logger.info("Completed Stage C v1 prototype in %.3f seconds", timer.seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
