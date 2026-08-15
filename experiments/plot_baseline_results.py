from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "outputs/experiments/stage_b_30stocks_baselines"
DEFAULT_OUTPUT = REPO_ROOT / "reports/figures"

MODELS = [
    "naive", "moving_average", "arima", "lstm",
    "minimalist_transformer", "frets", "timegnn",
]
TRAINED_MODELS = ["lstm", "minimalist_transformer", "frets", "timegnn"]
LABELS = {
    "naive": "Naive",
    "moving_average": "Moving average",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "minimalist_transformer": "Minimalist Transformer",
    "frets": "FreTS",
    "timegnn": "Time-GNN",
}
COLORS = {
    "naive": "#7f8c8d",
    "moving_average": "#f39c12",
    "arima": "#8e44ad",
    "lstm": "#2980b9",
    "minimalist_transformer": "#16a085",
    "frets": "#c0392b",
    "timegnn": "#2c3e50",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction_paths(results_root: Path, model: str) -> list[Path]:
    paths = sorted(results_root.glob(f"{model}_seed*/predictions.csv"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected three prediction files for {model}, found {len(paths)}")
    return paths


def prediction_series(results_root: Path, model: str, split: str) -> pd.DataFrame:
    frames = []
    for path in load_prediction_paths(results_root, model):
        frame = pd.read_csv(path, parse_dates=["target_date"])
        frame = frame[frame["split"] == split].copy()
        frame["seed_run"] = path.parent.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("target_date", as_index=False).agg(
        actual_return=("target_return", "mean"),
        prediction=("prediction", "mean"),
        stocks=("stock_code", "nunique"),
    )


def plot_predictions(results_root: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharey=True)
    for axis, split in zip(axes, ("validation", "test")):
        actual = prediction_series(results_root, MODELS[0], split)
        axis.plot(
            actual["target_date"], actual["actual_return"],
            color="black", linewidth=3.0, marker="o", markersize=5, label="Actual",
        )
        for model in MODELS:
            series = prediction_series(results_root, model, split)
            if not (series["stocks"] == 30).all():
                raise RuntimeError(f"{model}/{split} does not retain the 30-stock universe")
            axis.plot(
                series["target_date"], series["prediction"],
                color=COLORS[model], linewidth=1.8, marker=".", alpha=0.9,
                label=LABELS[model],
            )
        axis.axhline(0.0, color="#bdc3c7", linewidth=1.0)
        axis.set_title(f"Original {split} label — cross-sectional mean across 30 stocks")
        axis.set_ylabel("Next-week return")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(True, alpha=0.22)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x", rotation=25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.065),
        ncol=4, frameon=False,
    )
    fig.suptitle("Prediction comparison — unified 30-stock weekly experiment", fontsize=16, fontweight="bold")
    fig.text(
        0.5, 0.018,
        "Selection-exposed TRAIN evidence; original validation/test labels are not independent SCREENING/FINAL evidence.",
        ha="center", fontsize=10, color="#7f1d1d",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    path = output_dir / "prediction_comparison_30stocks.png"
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_histories(results_root: Path, model: str) -> list[pd.DataFrame]:
    paths = sorted(results_root.glob(f"{model}_seed*/training_history.json"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected three training histories for {model}, found {len(paths)}")
    return [pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def aligned_stat(histories: list[pd.DataFrame], column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maximum_epoch = max(int(history["epoch"].max()) for history in histories)
    epochs = np.arange(1, maximum_epoch + 1)
    values = np.full((len(histories), maximum_epoch), np.nan, dtype=float)
    for index, history in enumerate(histories):
        for row in history.itertuples(index=False):
            values[index, int(row.epoch) - 1] = float(getattr(row, column))
    valid = np.any(np.isfinite(values), axis=0)
    return (
        epochs[valid],
        np.nanmean(values[:, valid], axis=0),
        np.nanmin(values[:, valid], axis=0),
        np.nanmax(values[:, valid], axis=0),
    )


def plot_training_curves(results_root: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, model in zip(axes.flat, TRAINED_MODELS):
        histories = load_histories(results_root, model)
        for column, label, color in (
            ("train_loss", "Train loss", "#2563eb"),
            ("validation_loss", "Validation loss", "#dc2626"),
        ):
            epochs, mean, low, high = aligned_stat(histories, column)
            axis.plot(epochs, mean, color=color, linewidth=2.2, label=f"Mean {label.lower()}")
            axis.fill_between(epochs, low, high, color=color, alpha=0.16, label=f"Seed range: {label.lower()}")
        axis.set_title(f"{LABELS[model]} — 3 seeds")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Training-scale loss")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.22)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("Training curves — fixed 30-stock universe", fontsize=16, fontweight="bold")
    fig.text(
        0.5, 0.018,
        "Shaded bands show min–max across seeds. FreTS uses standardized target loss, so loss levels are compared within model, not across panels. Selection-exposed TRAIN.",
        ha="center", fontsize=9, color="#7f1d1d",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    path = output_dir / "training_curves_30stocks.png"
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prediction and training charts for Stage B baselines.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = plot_predictions(results_root, output_dir)
    training_path = plot_training_curves(results_root, output_dir)
    summary_path = results_root / "baseline_summary.csv"
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "SELECTION_EXPOSED_NON_INDEPENDENT",
        "stock_count": 30,
        "seeds": [20260723, 20260724, 20260725],
        "models": MODELS,
        "inputs": {
            "baseline_summary": str(summary_path),
            "baseline_summary_sha256": sha256(summary_path),
            "run_count": 21,
        },
        "figures": {
            prediction_path.name: sha256(prediction_path),
            training_path.name: sha256(training_path),
        },
        "screening_rows_read": 0,
        "final_rows_read": 0,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
