from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.bounded_ablations import minimalist_feature_view  # noqa: E402
from experiments.core import DataBundle, write_json  # noqa: E402
from stage_c.inference import LoadedFixedEnsemble  # noqa: E402


SEEDS = (20260723, 20260724, 20260725)
DATA_ROOT = REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1"
SOURCE_ROOT = REPO_ROOT / "outputs/experiments/stage_c_30stocks_graph_stabilization"
MODEL_ROOT = REPO_ROOT / "outputs/experiments/stage_c_30stocks_recommended_v2"
OUTPUT_ROOT = REPO_ROOT / "outputs/diagnostics/stage_c_recommended_v2"


def regression_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float | int]:
    y = frame["target_return"].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    error = prediction - y
    return {
        "samples": int(len(frame)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "direction_accuracy": float(np.mean((prediction > 0) == (y > 0))),
        "direction_f1": float(f1_score(y > 0, prediction > 0, zero_division=0)),
    }


def quantile_group(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=q, labels=labels)


def save_horizontal_bar(frame: pd.DataFrame, value: str, label: str, title: str, path: Path) -> None:
    ordered = frame.sort_values(value)
    height = max(5.0, len(ordered) * 0.24)
    figure, axis = plt.subplots(figsize=(9, height))
    colors = ["#2E86AB" if item <= ordered[value].median() else "#E07A5F" for item in ordered[value]]
    axis.barh(ordered[label], ordered[value], color=colors)
    axis.set(xlabel=value, title=title)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    data_started = time.perf_counter()
    data = minimalist_feature_view(DataBundle.load(DATA_ROOT), "price_only")
    data_load_seconds = time.perf_counter() - data_started
    stock_basic = pd.read_csv(DATA_ROOT / "stock_basic.csv.gz", usecols=["stock_code", "stock_name", "industry"])
    stock_basic = stock_basic.drop_duplicates("stock_code")

    all_samples = []
    engineering_rows = []
    for seed in SEEDS:
        manifest = MODEL_ROOT / f"fixed_control_ensemble_v2_seed{seed}" / "model_manifest.json"
        gc.collect()
        rss_before = process.memory_info().rss
        load_started = time.perf_counter()
        ensemble = LoadedFixedEnsemble(manifest, REPO_ROOT, device="cpu")
        load_seconds = time.perf_counter() - load_started
        rss_after_load = process.memory_info().rss

        # Warm-up is excluded from the timing distribution.
        ensemble.predict(data, "validation")
        timings = []
        prediction = None
        components = None
        for _ in range(10):
            started = time.perf_counter()
            prediction, components = ensemble.predict(data, "validation")
            timings.append(time.perf_counter() - started)
        assert prediction is not None and components is not None
        rss_after_predict = process.memory_info().rss

        samples = data.samples["validation"][[
            "stock_code", "trade_date", "target_date", "target_return", "target_direction",
        ]].copy()
        samples["seed"] = seed
        samples["ensemble_prediction"] = prediction
        samples["temporal_prediction"] = components["temporal_only"]
        samples["fixed_graph_prediction"] = components["fixed_temporal_graph"]
        samples["ensemble_error"] = samples["ensemble_prediction"] - samples["target_return"]
        samples["ensemble_abs_error"] = samples["ensemble_error"].abs()
        samples["temporal_abs_error"] = (samples["temporal_prediction"] - samples["target_return"]).abs()
        samples["fixed_graph_abs_error"] = (samples["fixed_graph_prediction"] - samples["target_return"]).abs()
        samples["component_disagreement"] = (
            samples["temporal_prediction"] - samples["fixed_graph_prediction"]
        ).abs()
        samples["component_sign_disagreement"] = (
            (samples["temporal_prediction"] > 0) != (samples["fixed_graph_prediction"] > 0)
        )
        samples["average_component_abs_error"] = 0.5 * (
            samples["temporal_abs_error"] + samples["fixed_graph_abs_error"]
        )
        samples["ensemble_gain_vs_average"] = (
            samples["average_component_abs_error"] - samples["ensemble_abs_error"]
        )
        samples["oracle_component_abs_error"] = samples[[
            "temporal_abs_error", "fixed_graph_abs_error",
        ]].min(axis=1)
        samples["ensemble_gap_vs_oracle"] = (
            samples["ensemble_abs_error"] - samples["oracle_component_abs_error"]
        )
        samples["ensemble_beats_both"] = samples["ensemble_abs_error"] < samples["oracle_component_abs_error"] - 1e-12
        samples["better_component"] = np.where(
            samples["temporal_abs_error"] < samples["fixed_graph_abs_error"],
            "temporal_only", "fixed_temporal_graph",
        )
        all_samples.append(samples)

        component_costs = []
        for component in ensemble.components:
            checkpoint = component.model.checkpoint_path
            source_metrics = json.loads((checkpoint.parent / "metrics.json").read_text(encoding="utf-8"))
            component_costs.append({
                "seed": seed,
                "scope": "component",
                "component": component.name,
                "parameters": sum(parameter.numel() for parameter in component.model.model.parameters()),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "training_run_seconds": float(source_metrics["duration_seconds"]),
                "load_seconds": None,
                "inference_mean_seconds_120": None,
                "inference_median_seconds_120": None,
                "inference_p95_seconds_120": None,
                "throughput_samples_per_second": None,
                "rss_delta_load_bytes": None,
                "rss_delta_total_bytes": None,
                "data_load_seconds_shared": data_load_seconds,
            })
        engineering_rows.extend(component_costs)
        total_parameters = sum(row["parameters"] for row in component_costs)
        total_checkpoint_bytes = sum(row["checkpoint_bytes"] for row in component_costs)
        total_training_seconds = sum(row["training_run_seconds"] for row in component_costs)
        median_inference = float(np.median(timings))
        engineering_rows.append({
            "seed": seed,
            "scope": "ensemble",
            "component": "fixed_control_ensemble_v2",
            "parameters": total_parameters,
            "checkpoint_bytes": total_checkpoint_bytes,
            "training_run_seconds": total_training_seconds,
            "load_seconds": load_seconds,
            "inference_mean_seconds_120": float(np.mean(timings)),
            "inference_median_seconds_120": median_inference,
            "inference_p95_seconds_120": float(np.percentile(timings, 95)),
            "throughput_samples_per_second": float(120 / median_inference),
            "rss_delta_load_bytes": int(max(0, rss_after_load - rss_before)),
            "rss_delta_total_bytes": int(max(0, rss_after_predict - rss_before)),
            "data_load_seconds_shared": data_load_seconds,
        })

    samples = pd.concat(all_samples, ignore_index=True).merge(
        stock_basic, on="stock_code", how="left", validate="many_to_one",
    )
    base_targets = samples[samples["seed"] == SEEDS[0]][["stock_code", "target_date", "target_return"]].copy()
    base_targets["return_quintile"] = quantile_group(
        base_targets["target_return"], 5,
        ["Q1 strongest decline", "Q2 decline", "Q3 near zero", "Q4 rise", "Q5 strongest rise"],
    ).astype(str)
    base_targets["magnitude_group"] = quantile_group(
        base_targets["target_return"].abs(), 3,
        ["Low absolute return", "Medium absolute return", "High absolute return"],
    ).astype(str)
    base_targets["direction_group"] = np.where(base_targets["target_return"] > 0, "Up", "Down_or_zero")
    samples = samples.merge(
        base_targets.drop(columns="target_return"),
        on=["stock_code", "target_date"], how="left", validate="many_to_one",
    )
    samples["disagreement_quartile"] = quantile_group(
        samples["component_disagreement"], 4,
        ["D1 low", "D2", "D3", "D4 high"],
    ).astype(str)

    stock_rows = []
    for (stock_code, stock_name, industry), frame in samples.groupby(
        ["stock_code", "stock_name", "industry"], dropna=False, sort=True,
    ):
        metrics = regression_metrics(frame, "ensemble_prediction")
        prediction_stability = frame.pivot_table(
            index="target_date", columns="seed", values="ensemble_prediction",
        ).std(axis=1, ddof=1).mean()
        stock_rows.append({
            "stock_code": stock_code,
            "stock_name": stock_name,
            "industry": industry,
            **metrics,
            "temporal_mae": float(frame["temporal_abs_error"].mean()),
            "fixed_graph_mae": float(frame["fixed_graph_abs_error"].mean()),
            "mae_improvement_vs_temporal_pct": float(
                (frame["temporal_abs_error"].mean() - frame["ensemble_abs_error"].mean())
                / frame["temporal_abs_error"].mean() * 100
            ),
            "mae_improvement_vs_fixed_pct": float(
                (frame["fixed_graph_abs_error"].mean() - frame["ensemble_abs_error"].mean())
                / frame["fixed_graph_abs_error"].mean() * 100
            ),
            "mean_component_disagreement": float(frame["component_disagreement"].mean()),
            "component_sign_disagreement_rate": float(frame["component_sign_disagreement"].mean()),
            "mean_cross_seed_prediction_std": float(prediction_stability),
        })
    stock_diagnostics = pd.DataFrame(stock_rows).sort_values("mae")

    group_rows = []
    for group_type, group_column in (
        ("return_quintile", "return_quintile"),
        ("direction", "direction_group"),
        ("absolute_return_magnitude", "magnitude_group"),
    ):
        for group, frame in samples.groupby(group_column, sort=False):
            group_rows.append({
                "group_type": group_type,
                "group": group,
                **regression_metrics(frame, "ensemble_prediction"),
                "mean_target_return": float(frame["target_return"].mean()),
                "temporal_mae": float(frame["temporal_abs_error"].mean()),
                "fixed_graph_mae": float(frame["fixed_graph_abs_error"].mean()),
                "mean_component_disagreement": float(frame["component_disagreement"].mean()),
                "ensemble_gain_vs_average": float(frame["ensemble_gain_vs_average"].mean()),
            })
    return_groups = pd.DataFrame(group_rows)

    disagreement_rows = []
    for bucket, frame in samples.groupby("disagreement_quartile", sort=False):
        disagreement_rows.append({
            "disagreement_quartile": bucket,
            "samples": len(frame),
            "mean_component_disagreement": float(frame["component_disagreement"].mean()),
            "ensemble_mae": float(frame["ensemble_abs_error"].mean()),
            "temporal_mae": float(frame["temporal_abs_error"].mean()),
            "fixed_graph_mae": float(frame["fixed_graph_abs_error"].mean()),
            "ensemble_gain_vs_average": float(frame["ensemble_gain_vs_average"].mean()),
            "ensemble_gap_vs_oracle": float(frame["ensemble_gap_vs_oracle"].mean()),
            "ensemble_beats_both_rate": float(frame["ensemble_beats_both"].mean()),
            "component_sign_disagreement_rate": float(frame["component_sign_disagreement"].mean()),
        })
    disagreement_buckets = pd.DataFrame(disagreement_rows)

    correlation_rows = []
    for seed, frame in samples.groupby("seed"):
        correlation_rows.append({
            "seed": seed,
            "temporal_fixed_prediction_correlation": float(
                frame["temporal_prediction"].corr(frame["fixed_graph_prediction"])
            ),
            "mean_absolute_disagreement": float(frame["component_disagreement"].mean()),
            "sign_disagreement_rate": float(frame["component_sign_disagreement"].mean()),
            "ensemble_gain_vs_average": float(frame["ensemble_gain_vs_average"].mean()),
            "ensemble_beats_both_rate": float(frame["ensemble_beats_both"].mean()),
        })
    component_summary = pd.DataFrame(correlation_rows)

    engineering_cost = pd.DataFrame(engineering_rows)
    engineering_summary = engineering_cost.groupby(["scope", "component"], as_index=False).agg(
        runs=("seed", "count"),
        parameters=("parameters", "mean"),
        checkpoint_bytes=("checkpoint_bytes", "mean"),
        training_seconds_mean=("training_run_seconds", "mean"),
        training_seconds_std=("training_run_seconds", "std"),
        load_seconds_mean=("load_seconds", "mean"),
        inference_median_seconds_120=("inference_median_seconds_120", "mean"),
        inference_p95_seconds_120=("inference_p95_seconds_120", "mean"),
        throughput_samples_per_second=("throughput_samples_per_second", "mean"),
        rss_delta_load_bytes=("rss_delta_load_bytes", "mean"),
        rss_delta_total_bytes=("rss_delta_total_bytes", "mean"),
        data_load_seconds_shared=("data_load_seconds_shared", "mean"),
    )

    samples.to_csv(OUTPUT_ROOT / "sample_diagnostics.csv", index=False, encoding="utf-8-sig")
    stock_diagnostics.to_csv(OUTPUT_ROOT / "per_stock_diagnostics.csv", index=False, encoding="utf-8-sig")
    return_groups.to_csv(OUTPUT_ROOT / "return_group_diagnostics.csv", index=False, encoding="utf-8-sig")
    disagreement_buckets.to_csv(OUTPUT_ROOT / "component_disagreement_buckets.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(OUTPUT_ROOT / "component_disagreement_summary.csv", index=False, encoding="utf-8-sig")
    engineering_cost.to_csv(OUTPUT_ROOT / "engineering_cost_by_seed.csv", index=False, encoding="utf-8-sig")
    engineering_summary.to_csv(OUTPUT_ROOT / "engineering_cost_summary.csv", index=False, encoding="utf-8-sig")
    samples.nlargest(20, "ensemble_gain_vs_average").to_csv(
        OUTPUT_ROOT / "top_ensemble_gain_samples.csv", index=False, encoding="utf-8-sig",
    )
    samples.nlargest(20, "ensemble_gap_vs_oracle").to_csv(
        OUTPUT_ROOT / "top_ensemble_oracle_gap_samples.csv", index=False, encoding="utf-8-sig",
    )

    save_horizontal_bar(
        stock_diagnostics, "mae", "stock_code", "Recommended v2 MAE by stock",
        OUTPUT_ROOT / "per_stock_mae.png",
    )
    quintiles = return_groups[return_groups["group_type"] == "return_quintile"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(quintiles["group"], quintiles["mae"], color="#2E86AB")
    axis.set(ylabel="MAE", title="Recommended v2 error by realized-return quintile")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "return_quintile_mae.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        disagreement_buckets["disagreement_quartile"],
        disagreement_buckets["ensemble_gain_vs_average"], color="#81B29A",
    )
    axis.axhline(0.0, color="#333333", linewidth=1)
    axis.set(ylabel="MAE gain vs average component error", title="Ensemble gain by component disagreement")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "disagreement_vs_gain.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        samples["temporal_prediction"], samples["fixed_graph_prediction"],
        c=samples["target_return"], cmap="coolwarm", alpha=0.55, s=18,
    )
    limits = [
        min(samples["temporal_prediction"].min(), samples["fixed_graph_prediction"].min()),
        max(samples["temporal_prediction"].max(), samples["fixed_graph_prediction"].max()),
    ]
    axis.plot(limits, limits, linestyle="--", color="#333333", linewidth=1)
    axis.set(
        xlabel="Temporal-only prediction", ylabel="Fixed-graph prediction",
        title="Component prediction disagreement",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "component_prediction_scatter.png", dpi=160)
    plt.close(figure)

    ensemble_cost = engineering_summary[engineering_summary["scope"] == "ensemble"].iloc[0]
    best_stocks = stock_diagnostics.head(5)[["stock_code", "stock_name", "mae"]].to_dict(orient="records")
    worst_stocks = stock_diagnostics.tail(5).sort_values("mae", ascending=False)[
        ["stock_code", "stock_name", "mae"]
    ].to_dict(orient="records")
    high_disagreement = disagreement_buckets.iloc[-1]
    summary: dict[str, Any] = {
        "scope": "recommended v2, 30-stock validation diagnostics, three frozen seeds",
        "test_status": "reserved_not_evaluated_or_used_for_selection",
        "engineering": {
            "total_parameters": int(ensemble_cost["parameters"]),
            "total_checkpoint_bytes": int(ensemble_cost["checkpoint_bytes"]),
            "mean_sequential_training_seconds": float(ensemble_cost["training_seconds_mean"]),
            "mean_model_load_seconds": float(ensemble_cost["load_seconds_mean"]),
            "mean_inference_median_seconds_120": float(ensemble_cost["inference_median_seconds_120"]),
            "mean_throughput_samples_per_second": float(ensemble_cost["throughput_samples_per_second"]),
            "mean_rss_delta_total_bytes": int(ensemble_cost["rss_delta_total_bytes"]),
            "shared_data_load_seconds": float(ensemble_cost["data_load_seconds_shared"]),
        },
        "component_disagreement": {
            "mean_prediction_correlation": float(component_summary["temporal_fixed_prediction_correlation"].mean()),
            "mean_absolute_disagreement": float(component_summary["mean_absolute_disagreement"].mean()),
            "mean_sign_disagreement_rate": float(component_summary["sign_disagreement_rate"].mean()),
            "mean_ensemble_gain_vs_average": float(component_summary["ensemble_gain_vs_average"].mean()),
            "high_disagreement_gain_vs_average": float(high_disagreement["ensemble_gain_vs_average"]),
            "high_disagreement_oracle_gap": float(high_disagreement["ensemble_gap_vs_oracle"]),
        },
        "best_stocks": best_stocks,
        "worst_stocks": worst_stocks,
    }
    write_json(OUTPUT_ROOT / "diagnostic_summary.json", summary)
    workbook_payload = {
        "summary": summary,
        "engineering_summary": json.loads(engineering_summary.to_json(orient="records", force_ascii=False)),
        "engineering_by_seed": json.loads(engineering_cost.to_json(orient="records", force_ascii=False)),
        "stock_diagnostics": json.loads(stock_diagnostics.to_json(orient="records", force_ascii=False)),
        "return_groups": json.loads(return_groups.to_json(orient="records", force_ascii=False)),
        "disagreement_buckets": json.loads(disagreement_buckets.to_json(orient="records", force_ascii=False)),
        "component_summary": json.loads(component_summary.to_json(orient="records", force_ascii=False)),
        "sample_diagnostics": json.loads(samples.to_json(orient="records", force_ascii=False, date_format="iso")),
    }
    write_json(OUTPUT_ROOT / "workbook_payload.json", workbook_payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
