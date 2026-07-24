from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.bounded_ablations import (  # noqa: E402
    FreTSBoundedAblationAdapter,
    MinimalistTransformerFeatureAblation,
)
from experiments.core import (  # noqa: E402
    Timer,
    create_logger,
    environment_info,
    evaluate_predictions,
    prediction_frame,
    set_global_seed,
    write_json,
)
from experiments.models import NaiveBaseline  # noqa: E402
from stage_c.trainer import GraphFrequencyExperiment  # noqa: E402
from stage_d.aggregation import aggregate_cross_fold  # noqa: E402
from stage_d.d2_baselines import (  # noqa: E402
    apply_fixed_shrinkage,
    build_fold_bundle,
    graph_price_bundle,
    load_locked_config,
    price_only_bundle,
    registered_models,
    shrinkage_model_name,
    validate_protocol,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(config: dict[str, Any], model_id: str, seed: int, data: Any) -> tuple[Any, Any, dict[str, Any]]:
    if model_id == "naive":
        return NaiveBaseline(), data, {"forecast_return": 0.0}
    if model_id == "frets_return_l4":
        model_config = dict(config["frets"])
        external = {"root": model_config.pop("root"), "model_file": model_config.pop("model_file")}
        model = FreTSBoundedAblationAdapter(model_config, external, seed, model_id)
        return model, data, model_config
    if model_id == "minimalist_price_only_l8":
        model_config = dict(config["minimalist_transformer"])
        model_config.pop("feature_set")
        model = MinimalistTransformerFeatureAblation(model_config, seed, model_id)
        return model, price_only_bundle(data), model_config
    if model_id in config["graph_overrides"]:
        base = json.loads(Path(config["graph_base_config_path"]).read_text(encoding="utf-8"))
        model_config = {**base, **config["graph_overrides"][model_id], "device": base.get("device", "auto")}
        model = GraphFrequencyExperiment(model_config, seed)
        model.name = model_id
        return model, graph_price_bundle(data), model_config
    raise ValueError(f"model is not preregistered: {model_id}")


def predict_model(model: Any, data: Any) -> np.ndarray:
    if isinstance(model, GraphFrequencyExperiment):
        prediction, _, _ = model.predict_with_diagnostics(data, "validation")
        return prediction
    return model.predict(data, "validation")


def metric_row(model_id: str, fold_id: str, seed: int, metrics: dict[str, Any]) -> dict[str, Any]:
    aggregate = metrics["aggregate"]
    return {
        "model": model_id,
        "fold_id": fold_id,
        "seed": int(seed),
        "samples": aggregate["samples"],
        "mae": aggregate["mae"],
        "rmse": aggregate["rmse"],
        "direction_accuracy": aggregate["direction_accuracy"],
        "direction_f1": aggregate["direction_f1"],
    }


def run(config_path: Path) -> dict[str, Any]:
    config = load_locked_config(config_path.resolve(), REPO_ROOT)
    protocol = validate_protocol(config)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "preregistered_config.json", config)
    write_json(output_root / "environment.json", environment_info(REPO_ROOT))
    rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    expected_base_models = list(config["base_models"])
    for fold in protocol["folds"]:
        fold_id = fold["fold_id"]
        data, fold_evidence = build_fold_bundle(config, protocol, fold_id, REPO_ROOT)
        for seed in config["seeds"]:
            base_predictions: dict[str, np.ndarray] = {}
            for model_id in expected_base_models:
                run_id = f"{fold_id}__{model_id}__seed{seed}"
                run_dir = output_root / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                logger = create_logger(run_dir)
                write_json(run_dir / "seeds.json", set_global_seed(int(seed)))
                started = datetime.now(timezone.utc).isoformat()
                logger.info("D-2 start fold=%s model=%s seed=%d", fold_id, model_id, seed)
                with Timer() as timer:
                    model, model_data, model_config = build_model(config, model_id, int(seed), data)
                    write_json(run_dir / "resolved_config.json", {
                        "experiment_id": config["experiment_id"],
                        "protocol_id": config["protocol_id"],
                        "protocol_sha256": config["protocol_sha256"],
                        "fold": fold,
                        "fold_evidence": fold_evidence,
                        "model": model_id,
                        "seed": int(seed),
                        "model_config": model_config,
                        "feature_columns": model_data.feature_columns,
                        "splits_used": ["train", "validation"],
                        "c4_rows_read": 0,
                        "future_d_screening_rows_read": 0,
                    })
                    model.fit(model_data, run_dir, logger)
                    prediction = predict_model(model, model_data)
                    base_predictions[model_id] = prediction
                    frame = prediction_frame(model_data.samples["validation"], prediction, "validation")
                    metrics = evaluate_predictions(frame)
                    frame.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
                    write_json(run_dir / "metrics.json", metrics)
                rows.append(metric_row(model_id, fold_id, int(seed), metrics))
                completed.append({
                    "run_id": run_id,
                    "fold_id": fold_id,
                    "model": model_id,
                    "seed": int(seed),
                    "started_at_utc": started,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": timer.seconds,
                    "prediction_sha256": _sha256(run_dir / "predictions.csv"),
                    "c4_rows_read": 0,
                    "future_d_screening_rows_read": 0,
                })
                logger.info("D-2 complete metrics=%s", metrics["aggregate"])
            samples = data.samples["validation"]
            for base_model in config["shrinkage"]["base_models"]:
                for alpha in config["shrinkage"]["alphas"]:
                    model_id = shrinkage_model_name(base_model, float(alpha))
                    prediction = apply_fixed_shrinkage(base_predictions[base_model], float(alpha))
                    frame = prediction_frame(samples, prediction, "validation")
                    metrics = evaluate_predictions(frame)
                    rows.append(metric_row(model_id, fold_id, int(seed), metrics))
                    derived_dir = output_root / "derived" / f"{fold_id}__{model_id}__seed{seed}"
                    derived_dir.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(derived_dir / "predictions.csv", index=False, encoding="utf-8-sig")
                    write_json(derived_dir / "metrics.json", metrics)
                    write_json(derived_dir / "derivation.json", {
                        "base_model": base_model,
                        "alpha": float(alpha),
                        "formula": config["shrinkage"]["formula"],
                        "naive_prediction": 0.0,
                        "post_result_candidate": False,
                    })
    metrics_frame = pd.DataFrame(rows).sort_values(["model", "fold_id", "seed"], ignore_index=True)
    expected_models = registered_models(config)
    expected_rows = len(expected_models) * len(protocol["folds"]) * len(config["seeds"])
    if len(metrics_frame) != expected_rows or set(metrics_frame["model"]) != set(expected_models):
        raise RuntimeError("D-2 model-fold-seed grid is incomplete")
    per_fold, summary, aggregate_metadata = aggregate_cross_fold(metrics_frame, baseline="naive")
    metrics_frame.to_csv(output_root / "metrics_by_fold_seed.csv", index=False, encoding="utf-8-sig")
    per_fold.to_csv(output_root / "per_fold_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "cross_fold_model_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_root / "completed_base_runs.json", completed)
    manifest = {
        **aggregate_metadata,
        "experiment_id": config["experiment_id"],
        "protocol_id": config["protocol_id"],
        "protocol_sha256": config["protocol_sha256"],
        "registered_model_count": len(expected_models),
        "expected_metric_rows": expected_rows,
        "actual_metric_rows": len(metrics_frame),
        "base_training_runs": len(completed),
        "fold_definition_mutated": False,
        "original_static_splits_used": False,
        "c4_rows_read": 0,
        "future_d_screening_rows_read": 0,
        "independent_screening": False,
        "warning": "D-2 is selection-exposed rolling-development evidence, not independent screening.",
    }
    write_json(output_root / "evidence_manifest.json", manifest)
    hashes = {}
    for name in (
        "preregistered_config.json", "metrics_by_fold_seed.csv", "per_fold_summary.csv",
        "cross_fold_model_summary.csv", "completed_base_runs.json", "evidence_manifest.json",
    ):
        hashes[name] = _sha256(output_root / name)
    write_json(output_root / "sha256_manifest.json", hashes)
    return {"output_root": str(output_root), "manifest": manifest, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered Stage D-2 bounded baselines.")
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "stage_d/configs/d2_baselines.json",
    )
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
