from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.core import (
    DataBundle, Timer, create_logger, environment_info, evaluate_predictions,
    prediction_frame, set_global_seed, write_json,
)
from experiments.models import build_model


def run_model(config: dict[str, Any], model_name: str, seed: int, repo_root: Path) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{model_name}_seed{seed}"
    output_dir = Path(config["output_root"]) / config["experiment_name"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir)
    resolved = dict(config)
    resolved.update({"selected_model": model_name, "selected_seed": seed, "run_id": run_id})
    write_json(output_dir / "resolved_config.json", resolved)
    write_json(output_dir / "environment.json", environment_info(repo_root))
    write_json(output_dir / "seeds.json", set_global_seed(seed))
    logger.info("Starting model=%s seed=%d", model_name, seed)

    with Timer() as timer:
        data = DataBundle.load(Path(config["data_root"]))
        logger.info(
            "Loaded train=%d validation=%d test=%d features=%d",
            len(data.samples["train"]), len(data.samples["validation"]),
            len(data.samples["test"]), len(data.feature_columns),
        )
        model = build_model(model_name, config, seed)
        model.fit(data, output_dir, logger)
        all_predictions = []
        metrics: dict[str, Any] = {}
        for split in ("validation", "test"):
            prediction = model.predict(data, split)
            frame = prediction_frame(data.samples[split], prediction, split)
            all_predictions.append(frame)
            metrics[split] = evaluate_predictions(frame)
            logger.info("%s metrics: %s", split, metrics[split]["aggregate"])
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    result = {
        "run_id": run_id,
        "model": model_name,
        "seed": seed,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": timer.seconds,
        "sample_counts": {split: len(frame) for split, frame in data.samples.items()},
        "feature_columns": data.feature_columns,
        "metrics": metrics,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "metrics.json", result)
    logger.info("Completed in %.3f seconds", timer.seconds)
    return result
