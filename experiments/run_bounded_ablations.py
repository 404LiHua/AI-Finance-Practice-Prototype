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

from experiments.bounded_ablations import (  # noqa: E402
    FreTSBoundedAblationAdapter,
    MinimalistTransformerFeatureAblation,
    minimalist_feature_view,
)
from experiments.core import (  # noqa: E402
    DataBundle,
    Timer,
    create_logger,
    environment_info,
    evaluate_predictions,
    load_config,
    prediction_frame,
    set_global_seed,
    write_json,
)


EVIDENCE_CLASS = "SELECTION_EXPOSED_TRAIN_NON_INDEPENDENT"


def build_variant(
    config: dict[str, Any], variant_id: str, seed: int, data: DataBundle,
) -> tuple[Any, DataBundle, dict[str, Any]]:
    variant = dict(config["variants"][variant_id])
    family = variant.pop("family")
    if family == "frets":
        model_config = {**config["frets_base"], **variant}
        model = FreTSBoundedAblationAdapter(
            model_config, config["upstream"]["frets"], seed, variant_id,
        )
        return model, data, model_config
    if family == "minimalist_transformer":
        feature_set = str(variant.pop("feature_set"))
        model_config = {**config["minimalist_transformer_base"], **variant}
        view = minimalist_feature_view(data, feature_set)
        model = MinimalistTransformerFeatureAblation(model_config, seed, variant_id)
        model_config = {**model_config, "feature_set": feature_set}
        return model, view, model_config
    raise ValueError(f"Unsupported ablation family: {family}")


def run_variant(config: dict[str, Any], variant_id: str, seed: int) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{variant_id}_seed{seed}"
    output_dir = Path(config["output_root"]) / config["experiment_name"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(output_dir)
    write_json(output_dir / "environment.json", environment_info(REPO_ROOT))
    write_json(output_dir / "seeds.json", set_global_seed(seed))
    logger.info("Starting bounded ablation variant=%s seed=%d", variant_id, seed)

    with Timer() as timer:
        data = DataBundle.load(Path(config["data_root"]))
        stock_count = int(data.panel["stock_code"].nunique())
        if stock_count != 30:
            raise RuntimeError(f"Bounded ablations require exactly 30 stocks, found {stock_count}")
        model, model_data, model_config = build_variant(config, variant_id, seed, data)
        resolved = {
            **config,
            "selected_variant": variant_id,
            "selected_seed": seed,
            "run_id": run_id,
            "resolved_model_config": model_config,
            "feature_columns": model_data.feature_columns,
            "evidence_class": EVIDENCE_CLASS,
            "screening_rows_read": 0,
            "final_rows_read": 0,
        }
        write_json(output_dir / "resolved_config.json", resolved)
        logger.info(
            "Loaded identical 30-stock samples train=%d validation=%d test=%d features=%d",
            len(model_data.samples["train"]), len(model_data.samples["validation"]),
            len(model_data.samples["test"]), len(model_data.feature_columns),
        )
        model.fit(model_data, output_dir, logger)
        prediction_frames = []
        metrics: dict[str, Any] = {}
        for split in ("validation", "test"):
            prediction = model.predict(model_data, split)
            frame = prediction_frame(model_data.samples[split], prediction, split)
            prediction_frames.append(frame)
            metrics[split] = evaluate_predictions(frame)
            logger.info("%s metrics: %s", split, metrics[split]["aggregate"])

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False, encoding="utf-8-sig",
    )
    result = {
        "run_id": run_id,
        "variant": variant_id,
        "family": config["variants"][variant_id]["family"],
        "seed": seed,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": timer.seconds,
        "stock_count": 30,
        "sample_counts": {split: len(frame) for split, frame in model_data.samples.items()},
        "feature_columns": model_data.feature_columns,
        "metrics": metrics,
        "output_dir": str(output_dir),
        "evidence_class": EVIDENCE_CLASS,
        "independent_evidence": False,
        "screening_rows_read": 0,
        "final_rows_read": 0,
    }
    write_json(output_dir / "metrics.json", result)
    logger.info("Completed bounded ablation in %.3f seconds", timer.seconds)
    return result


def aggregate(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    rows = []
    for path in sorted(experiment_dir.glob("*_seed*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("evidence_class") != EVIDENCE_CLASS:
            continue
        results.append(result)
        for split in ("validation", "test"):
            rows.append({
                "run_id": result["run_id"],
                "variant": result["variant"],
                "family": result["family"],
                "seed": result["seed"],
                "split": split,
                "stock_count": result["stock_count"],
                "evidence_class": result["evidence_class"],
                **result["metrics"][split]["aggregate"],
            })
    table = pd.DataFrame(rows).sort_values(["split", "mae", "variant", "seed"])
    summary = table.groupby(["family", "variant", "split"], as_index=False).agg(
        runs=("seed", "count"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_f1_mean=("direction_f1", "mean"),
    )
    table.to_csv(experiment_dir / "ablation_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(experiment_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    write_json(experiment_dir / "completed_runs.json", results)
    write_json(experiment_dir / "evidence_manifest.json", {
        "evidence_class": EVIDENCE_CLASS,
        "independent_evidence": False,
        "stock_count": 30,
        "run_count": len(results),
        "screening_rows_read": 0,
        "final_rows_read": 0,
        "warning": "Validation and original test labels are selection-exposed TRAIN evidence only.",
    })
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded TRAIN-only Stage B ablations.")
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "experiments/configs/bounded_ablations.json",
    )
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config.resolve(), REPO_ROOT)
    variants = args.variants or list(config["variants"])
    unknown = sorted(set(variants) - set(config["variants"]))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    seeds = args.seeds or config["seeds"]
    results = [run_variant(config, variant, int(seed)) for variant in variants for seed in seeds]
    experiment_dir = Path(config["output_root"]) / config["experiment_name"]
    _, summary = aggregate(experiment_dir)
    print(json.dumps({
        "new_runs": len(results),
        "output_dir": str(experiment_dir),
        "evidence_class": EVIDENCE_CLASS,
    }, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
