from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import DataBundle, evaluate_predictions, prediction_frame, write_json  # noqa: E402
from stage_c.inference import LoadedFixedEnsemble, LoadedStageCComponent, sha256_file  # noqa: E402


SEEDS = (20260723, 20260724, 20260725)
SCREENING_START = pd.Timestamp("2023-06-09")
SCREENING_END = pd.Timestamp("2024-06-07")
DATA_ROOT = REPO_ROOT / "data/screening/baostock_30stocks_20221125_20240607"
OUTPUT_ROOT = REPO_ROOT / "outputs/screening/stage_c_recommended_v2_c4_20230609_20240607"
FREEZE_CONFIG = REPO_ROOT / "stage_c/configs/recommended_v2_freeze_c3.json"
FREEZE_RECEIPT = REPO_ROOT / "stage_c/frozen/recommended_v2_c3/FREEZE_RECEIPT.json"
RECOMMENDED_ROOT = REPO_ROOT / "outputs/experiments/stage_c_30stocks_recommended_v2"
STABILIZATION_ROOT = REPO_ROOT / "outputs/experiments/stage_c_30stocks_graph_stabilization"
BOUNDED_ROOT = REPO_ROOT / "outputs/experiments/stage_b_30stocks_bounded_ablations"
GRAPH_V1_ROOT = REPO_ROOT / "outputs/experiments/stage_c_graph_frequency_v1"
DEVELOPMENT_DATA_ROOT = REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1"
FRETS_SOURCE = Path("D:/项目/源文件/deploy/FreTS-main/models/FreTS.py")


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_selected_stocks() -> list[str]:
    return [
        line.strip()
        for line in (DEVELOPMENT_DATA_ROOT / "selected_stocks.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_screening_bundle() -> tuple[DataBundle, dict[str, Any]]:
    codes = read_selected_stocks()
    frames = []
    source_files = []
    for code in codes:
        path = DATA_ROOT / f"{code}.weekly_qfq.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing SCREENING source file: {path}")
        frame = pd.read_csv(path)
        frame = frame.rename(columns={
            "project_stock_code": "stock_code",
            "date": "trade_date",
            "open": "model_open",
            "high": "model_high",
            "low": "model_low",
            "close": "model_close",
            "volume": "model_volume_hands",
            "amount": "model_amount_thousand_cny",
        })
        frame["stock_code"] = code
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise")
        numeric = [
            "model_open", "model_high", "model_low", "model_close",
            "model_volume_hands", "model_amount_thousand_cny",
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[numeric].isna().any().any():
            raise ValueError(f"non-numeric SCREENING prices in {path}")
        frames.append(frame[["stock_code", "trade_date", *numeric]])
        source_files.append({
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": len(frame),
        })

    panel = pd.concat(frames, ignore_index=True).sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    if panel.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("duplicate stock-date rows in SCREENING source")
    grouped = panel.groupby("stock_code", sort=False, group_keys=False)
    panel["return_1w"] = grouped["model_close"].pct_change(fill_method=None)
    panel["log_return_1w"] = grouped["model_close"].transform(lambda values: np.log(values).diff())
    previous_close = grouped["model_close"].shift(1)
    panel["intraweek_range"] = (
        (panel["model_high"] - panel["model_low"]) / previous_close.replace(0, np.nan)
    )
    panel["candle_body"] = (
        (panel["model_close"] - panel["model_open"]) / panel["model_open"].replace(0, np.nan)
    )
    for window in (4, 12, 26):
        panel[f"close_ma_{window}"] = grouped["model_close"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).mean()
        )
        panel[f"return_vol_{window}"] = grouped["return_1w"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).std()
        )
    panel["calendar_week_end"] = panel["trade_date"]
    panel["target_close"] = grouped["model_close"].shift(-1)
    panel["target_date"] = grouped["trade_date"].shift(-1)
    panel["target_return"] = panel["target_close"] / panel["model_close"] - 1.0
    panel["target_direction"] = np.where(
        panel["target_return"].notna(), (panel["target_return"] > 0).astype(float), np.nan
    )
    panel["split"] = "context"
    scoring_mask = (
        panel["trade_date"].ge(SCREENING_START)
        & panel["target_date"].le(SCREENING_END)
        & panel["target_return"].notna()
    )
    panel.loc[scoring_mask, "split"] = "screening"
    panel["sample_eligible"] = scoring_mask
    samples = panel.loc[scoring_mask].reset_index(drop=True)

    stock_counts = samples.groupby("stock_code").size()
    if list(sorted(stock_counts.index)) != list(sorted(codes)):
        raise ValueError("SCREENING stock scope does not match frozen 30-stock universe")
    if int(stock_counts.min()) < 4:
        raise ValueError("SCREENING has fewer than four eligible samples for at least one stock")
    date_counts = samples.groupby("trade_date")["stock_code"].nunique()
    if not date_counts.eq(len(codes)).all():
        raise ValueError("SCREENING does not have identical cross-sectional rows for all stocks")

    feature_columns = [
        "model_open", "model_high", "model_low", "model_close",
        "return_1w", "log_return_1w", "intraweek_range", "candle_body",
        "close_ma_4", "close_ma_12", "close_ma_26",
        "return_vol_4", "return_vol_12", "return_vol_26",
    ]
    bundle = DataBundle(panel=panel, samples={"screening": samples}, feature_columns=feature_columns)
    manifest = {
        "provider": "BaoStock",
        "price_adjustment": "forward-adjusted",
        "frequency": "weekly",
        "context_start": panel["trade_date"].min().date().isoformat(),
        "screening_start": SCREENING_START.date().isoformat(),
        "screening_end": SCREENING_END.date().isoformat(),
        "last_scored_trade_date": samples["trade_date"].max().date().isoformat(),
        "stock_count": len(codes),
        "screening_rows": len(samples),
        "samples_per_stock_min": int(stock_counts.min()),
        "samples_per_stock_max": int(stock_counts.max()),
        "source_manifest": (DATA_ROOT / "manifest.json").relative_to(REPO_ROOT).as_posix(),
        "source_manifest_sha256": sha256_file(DATA_ROOT / "manifest.json"),
        "source_files": source_files,
    }
    manifest["source_set_sha256"] = canonical_sha256(source_files)
    return bundle, manifest


def build_sequences(
    data: DataBundle,
    split: str,
    feature_columns: list[str],
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    values = data.panel[feature_columns].to_numpy(dtype=np.float32)
    values = np.where(np.isfinite(values), values, np.asarray(medians, dtype=np.float32))
    values = (values - np.asarray(means, dtype=np.float32)) / np.asarray(stds, dtype=np.float32)
    scaled = data.panel[["stock_code", "trade_date"]].copy()
    scaled["vector"] = list(values.astype(np.float32))
    lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
    for stock_code, frame in scaled.groupby("stock_code", sort=True):
        vectors = frame["vector"].tolist()
        for index, date in enumerate(frame["trade_date"].tolist()):
            sequence = vectors[max(0, index - sequence_length + 1):index + 1]
            if len(sequence) < sequence_length:
                sequence = [sequence[0]] * (sequence_length - len(sequence)) + sequence
            lookup[(stock_code, date)] = np.stack(sequence)
    return np.stack([
        lookup[(row.stock_code, row.trade_date)]
        for row in data.samples[split].itertuples(index=False)
    ]).astype(np.float32)


class MinimalistNetwork(nn.Module):
    def __init__(self, input_size: int, sequence_length: int, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config["nhead"]),
            dim_feedforward=int(config["dim_feedforward"]),
            dropout=float(config["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(config["num_layers"]), enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(values) + self.position[:, :values.shape[1]]
        encoded = self.encoder(encoded)
        return self.head(self.output_norm(encoded[:, -1])).squeeze(-1)


def predict_minimalist(checkpoint_path: Path, data: DataBundle, split: str) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    columns = list(checkpoint["feature_columns"])
    sequence_length = int(config["sequence_length"])
    model = MinimalistNetwork(len(columns), sequence_length, config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    values = build_sequences(
        data, split, columns, checkpoint["medians"], checkpoint["means"],
        checkpoint["stds"], sequence_length,
    )
    with torch.no_grad():
        result = model(torch.from_numpy(values)).numpy().astype(float)
    if not np.isfinite(result).all():
        raise FloatingPointError("Minimalist Transformer produced non-finite predictions")
    return result


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("stage_c_c4_frets", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load FreTS source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_frets(checkpoint_path: Path, data: DataBundle, split: str) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if sha256_file(FRETS_SOURCE) != checkpoint["source_sha256"]:
        raise ValueError("FreTS source SHA-256 does not match frozen checkpoint")
    config = dict(checkpoint["adapter_config"])
    columns = list(checkpoint["feature_columns"])
    sequence_length = int(config["sequence_length"])
    module = load_module(FRETS_SOURCE)
    model = module.Model(SimpleNamespace(
        pred_len=1,
        enc_in=len(columns),
        seq_len=sequence_length,
        channel_independence=str(config.get("channel_independence", "0")),
    ))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    values = build_sequences(
        data, split, columns, checkpoint["medians"], checkpoint["means"],
        checkpoint["stds"], sequence_length,
    )
    predictions = []
    batch_size = int(config["batch_size"])
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size])
            predictions.append(model(batch).reshape(batch.shape[0], -1)[:, 0].numpy())
    result = np.concatenate(predictions).astype(float)
    result = result * float(checkpoint["target_std"]) + float(checkpoint["target_mean"])
    if not np.isfinite(result).all():
        raise FloatingPointError("FreTS produced non-finite predictions")
    return result


def reference_validation_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "split" in frame:
        frame = frame[frame["split"] == "validation"].copy()
    return frame.reset_index(drop=True)


def verify_frozen_loaders() -> list[dict[str, Any]]:
    development = DataBundle.load(DEVELOPMENT_DATA_ROOT)
    checks = []
    for seed in SEEDS:
        frets_path = BOUNDED_ROOT / f"frets_return_l4_seed{seed}/model.pt"
        frets_prediction = predict_frets(frets_path, development, "validation")
        frets_reference = reference_validation_frame(
            BOUNDED_ROOT / f"frets_return_l4_seed{seed}/predictions.csv"
        )["prediction"].to_numpy(dtype=float)
        checks.append({
            "model": "frets_return_l4", "seed": seed,
            "max_abs_difference": float(np.max(np.abs(frets_prediction - frets_reference))),
        })

        minimalist_path = BOUNDED_ROOT / f"minimalist_price_only_l8_seed{seed}/model.pt"
        minimalist_prediction = predict_minimalist(minimalist_path, development, "validation")
        minimalist_reference = reference_validation_frame(
            BOUNDED_ROOT / f"minimalist_price_only_l8_seed{seed}/predictions.csv"
        )["prediction"].to_numpy(dtype=float)
        checks.append({
            "model": "minimalist_price_only_l8", "seed": seed,
            "max_abs_difference": float(np.max(np.abs(minimalist_prediction - minimalist_reference))),
        })

        graph_path = GRAPH_V1_ROOT / f"graph_frequency_v1_seed{seed}/model.pt"
        graph_component = LoadedStageCComponent(graph_path)
        graph_view = DataBundle(
            panel=development.panel,
            samples=development.samples,
            feature_columns=graph_component.feature_columns,
        )
        graph_prediction = graph_component.predict(graph_view, "validation")
        graph_reference = reference_validation_frame(
            GRAPH_V1_ROOT / f"graph_frequency_v1_seed{seed}/predictions.csv"
        )["prediction"].to_numpy(dtype=float)
        checks.append({
            "model": "graph_frequency_v1", "seed": seed,
            "max_abs_difference": float(np.max(np.abs(graph_prediction - graph_reference))),
        })
    maximum = max(item["max_abs_difference"] for item in checks)
    if maximum > 1e-7:
        raise ValueError(f"frozen baseline reconstruction failed: maximum difference={maximum}")
    return checks


def candidate_diagnostics(candidate_rows: pd.DataFrame) -> dict[str, float]:
    mean_by_row = candidate_rows.groupby(["stock_code", "trade_date", "target_date"], as_index=False).agg(
        target_return=("target_return", "first"),
        prediction=("prediction", "mean"),
    )
    mean_by_row["absolute_error"] = (mean_by_row["prediction"] - mean_by_row["target_return"]).abs()
    ranks = mean_by_row["target_return"].abs().rank(method="first")
    mean_by_row["magnitude_group"] = pd.qcut(ranks, q=3, labels=["low", "medium", "high"])
    aggregate_mae = float(mean_by_row["absolute_error"].mean())
    high_mae = float(mean_by_row.loc[mean_by_row["magnitude_group"] == "high", "absolute_error"].mean())
    stock_errors = mean_by_row.groupby("stock_code")["absolute_error"].sum().sort_values(ascending=False)
    return {
        "high_absolute_return_mae": high_mae,
        "high_absolute_return_mae_ratio": high_mae / aggregate_mae,
        "worst_five_stock_error_share": float(stock_errors.head(5).sum() / stock_errors.sum()),
    }


def decide(summary: pd.DataFrame, candidate_rows: pd.DataFrame) -> dict[str, Any]:
    indexed = summary.set_index("model")
    candidate = indexed.loc["fixed_control_ensemble_v2"]
    core = indexed.loc[["naive", "frets_return_l4", "minimalist_price_only_l8"]]
    best_core_mae = float(core["mae_mean"].min())
    best_core_rmse = float(core["rmse_mean"].min())
    naive_mae = float(indexed.loc["naive", "mae_mean"])
    mae_cv = float(candidate["mae_std"] / candidate["mae_mean"])
    diagnostics = candidate_diagnostics(candidate_rows)
    pass_checks = {
        "candidate_mae_le_best_core_baseline": float(candidate["mae_mean"]) <= best_core_mae,
        "candidate_rmse_within_5pct_of_best_core_baseline": float(candidate["rmse_mean"]) <= 1.05 * best_core_rmse,
        "direction_accuracy_floor": float(candidate["direction_accuracy_mean"]) >= 0.50,
        "direction_f1_floor": float(candidate["direction_f1_mean"]) >= 0.15,
        "three_seed_mae_cv_ceiling": mae_cv <= 0.10,
        "high_absolute_return_mae_ratio_ceiling": diagnostics["high_absolute_return_mae_ratio"] <= 2.25,
        "worst_five_stock_error_share_ceiling": diagnostics["worst_five_stock_error_share"] <= 0.45,
    }
    failure_checks = {
        "candidate_mae_worse_than_naive": float(candidate["mae_mean"]) > naive_mae,
        "candidate_rmse_more_than_10pct_worse_than_best_core_baseline": float(candidate["rmse_mean"]) > 1.10 * best_core_rmse,
        "direction_accuracy_below_failure_floor": float(candidate["direction_accuracy_mean"]) < 0.45,
        "direction_f1_below_failure_floor": float(candidate["direction_f1_mean"]) < 0.10,
        "three_seed_mae_cv_above_failure_ceiling": mae_cv > 0.15,
        "high_absolute_return_mae_ratio_above_failure_ceiling": diagnostics["high_absolute_return_mae_ratio"] > 2.50,
        "worst_five_stock_error_share_above_failure_ceiling": diagnostics["worst_five_stock_error_share"] > 0.50,
    }
    verdict = "FAIL" if any(failure_checks.values()) else "PASS" if all(pass_checks.values()) else "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "candidate": {
            "mae_mean": float(candidate["mae_mean"]),
            "rmse_mean": float(candidate["rmse_mean"]),
            "direction_accuracy_mean": float(candidate["direction_accuracy_mean"]),
            "direction_f1_mean": float(candidate["direction_f1_mean"]),
            "three_seed_mae_cv": mae_cv,
            **diagnostics,
        },
        "comparators": {
            "best_core_mae": best_core_mae,
            "best_core_rmse": best_core_rmse,
            "naive_mae": naive_mae,
        },
        "pass_checks": pass_checks,
        "failure_checks": failure_checks,
        "post_screening_repair_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-shot frozen Stage C independent SCREENING.")
    parser.add_argument("--verify-loaders-only", action="store_true")
    args = parser.parse_args()
    freeze = json.loads(FREEZE_CONFIG.read_text(encoding="utf-8"))
    receipt = json.loads(FREEZE_RECEIPT.read_text(encoding="utf-8"))
    if freeze["freeze_status"] != "LOCKED_BEFORE_SCREENING":
        raise ValueError("candidate freeze is not locked")
    if receipt["freeze_id"] != freeze["freeze_id"]:
        raise ValueError("freeze receipt does not match freeze policy")

    loader_checks = verify_frozen_loaders()
    if args.verify_loaders_only:
        print(json.dumps(loader_checks, ensure_ascii=False, indent=2))
        return

    bundle, data_manifest = build_screening_bundle()
    all_rows = []
    metric_rows = []
    for seed in SEEDS:
        manifest = RECOMMENDED_ROOT / f"fixed_control_ensemble_v2_seed{seed}/model_manifest.json"
        candidate = LoadedFixedEnsemble(manifest, REPO_ROOT)
        candidate_prediction, components = candidate.predict(bundle, "screening")
        predictions = {
            "fixed_control_ensemble_v2": candidate_prediction,
            "temporal_only_control": components["temporal_only"],
            "fixed_temporal_graph_control": components["fixed_temporal_graph"],
            "naive": np.zeros(len(candidate_prediction), dtype=float),
            "frets_return_l4": predict_frets(
                BOUNDED_ROOT / f"frets_return_l4_seed{seed}/model.pt", bundle, "screening"
            ),
            "minimalist_price_only_l8": predict_minimalist(
                BOUNDED_ROOT / f"minimalist_price_only_l8_seed{seed}/model.pt", bundle, "screening"
            ),
            "graph_frequency_v1": LoadedStageCComponent(
                GRAPH_V1_ROOT / f"graph_frequency_v1_seed{seed}/model.pt"
            ).predict(bundle, "screening"),
        }
        expected_length = len(bundle.samples["screening"])
        if any(len(values) != expected_length or not np.isfinite(values).all() for values in predictions.values()):
            raise ValueError("candidate or baseline produced invalid SCREENING predictions")
        for model_name, values in predictions.items():
            frame = prediction_frame(bundle.samples["screening"], values, "screening")
            frame.insert(1, "model", model_name)
            frame.insert(2, "seed", seed)
            all_rows.append(frame)
            aggregate = evaluate_predictions(frame)["aggregate"]
            metric_rows.append({"model": model_name, "seed": seed, **aggregate})

    predictions = pd.concat(all_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby("model", as_index=False).agg(
        runs=("seed", "count"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_accuracy_std=("direction_accuracy", "std"),
        direction_f1_mean=("direction_f1", "mean"),
        direction_f1_std=("direction_f1", "std"),
    ).sort_values("mae_mean")
    candidate_rows = predictions[predictions["model"] == "fixed_control_ensemble_v2"].copy()
    decision = decide(summary, candidate_rows)
    decision.update({
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_root_sha256": receipt["manifest_root_sha256"],
        "screening_split": "future_weekly_screening_20230609_20240607_v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_or_tuning_performed": False,
        "loader_reconstruction_max_abs_difference": max(
            item["max_abs_difference"] for item in loader_checks
        ),
    })

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(OUTPUT_ROOT / "screening_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(OUTPUT_ROOT / "screening_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "screening_model_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(loader_checks).to_csv(
        OUTPUT_ROOT / "loader_reconstruction_checks.csv", index=False, encoding="utf-8-sig"
    )
    write_json(OUTPUT_ROOT / "screening_data_manifest.json", data_manifest)
    write_json(OUTPUT_ROOT / "screening_decision.json", decision)
    evidence_files = []
    for path in sorted(OUTPUT_ROOT.iterdir()):
        if path.is_file():
            evidence_files.append({
                "file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)
            })
    write_json(OUTPUT_ROOT / "screening_evidence_manifest.json", {
        "freeze_id": freeze["freeze_id"],
        "screening_split": decision["screening_split"],
        "files": evidence_files,
        "evidence_set_sha256": canonical_sha256(evidence_files),
    })
    print(summary.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
