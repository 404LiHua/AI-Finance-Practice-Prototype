from __future__ import annotations

"""Metric-blind GPU training for the frozen WP24 shared H4/T2 backbone."""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FOLDS = tuple(f"REV2_RO_{i:02d}" for i in range(1, 7))
BAN = ("fresh", "screening", "final", "sealed_holdout")
RG3_DAILY = [
    "momentum_20d", "momentum_60d", "momentum_120d", "realized_volatility_20d",
    "realized_volatility_60d", "downside_volatility_60d", "current_drawdown_60d",
    "rsi_14", "macd_scaled", "bollinger_position_20", "amihud_20d",
    "zero_volume_fraction_20d", "volume_ratio_20d_60d", "intraday_range_mean_20d",
]
RG3_STRUCTURAL = [
    "log_market_cap_total", "log_market_cap_float", "float_share_ratio", "listing_age_weeks",
    "is_special_treatment", "is_suspended", "is_delisted_asof", "is_suspended_listing_asof",
    "history_weeks_scaled", "market_cap_small", "market_cap_medium", "market_cap_large",
    "structural_state_missing",
]
RG2_FEATURES = [
    "capital_event_this_week", "capital_event_increase_flag", "capital_event_decrease_flag",
    "log_total_shares_change_at_event", "log_tradable_shares_change_at_event",
    "tradable_share_ratio_change_at_event", "capital_event_age_260_scaled",
    "capital_history_missing_flag", "market_tradable_fraction", "market_eligible_fraction",
    "market_small_cap_fraction", "industry_tradable_fraction", "industry_eligible_fraction",
    "log1p_industry_member_count", "graph_mean_absolute_change",
    "graph_intra_industry_weight_fraction", "graph_mean_nonself_out_degree_scaled",
    "graph_max_nonself_out_degree_scaled",
]
FEATURES = [*RG3_DAILY, *RG3_STRUCTURAL, *RG2_FEATURES]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SharedMultiTask(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(96, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(64, 32), nn.SiLU(),
        )
        self.regression = nn.Linear(32, 2)
        self.ordinal_location = nn.Linear(32, 1)
        self.threshold_one = nn.Parameter(torch.tensor(-0.15))
        self.threshold_gap_raw = nn.Parameter(torch.tensor(0.30))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        regression = self.regression(hidden)
        mu = regression[:, 0]
        logvar = regression[:, 1].clamp(-4.0, 2.0)
        eta = self.ordinal_location(hidden).squeeze(1)
        threshold_one = self.threshold_one
        threshold_two = threshold_one + torch.nn.functional.softplus(self.threshold_gap_raw) + 0.05
        return mu, logvar, eta, torch.stack((threshold_one, threshold_two))


def probabilities(eta: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    first = torch.sigmoid(thresholds[0] - eta)
    second = torch.sigmoid(thresholds[1] - eta)
    output = torch.stack((first, second - first, 1.0 - second), dim=1).clamp_min(1e-7)
    return output / output.sum(dim=1, keepdim=True)


def loss_fn(mu: torch.Tensor, logvar: torch.Tensor, eta: torch.Tensor,
            thresholds: torch.Tensor, y_return: torch.Tensor, y_label: torch.Tensor) -> torch.Tensor:
    precision = torch.exp(-logvar)
    regression_nll = 0.5 * (precision * (y_return - mu).square() + logvar).mean()
    cdf_one_logit = thresholds[0] - eta
    cdf_two_logit = thresholds[1] - eta
    first = (y_label == 0).float()
    second = (y_label <= 1).float()
    ordinal = 0.5 * (
        torch.nn.functional.binary_cross_entropy_with_logits(cdf_one_logit, first)
        + torch.nn.functional.binary_cross_entropy_with_logits(cdf_two_logit, second)
    )
    return regression_nll + 0.5 * ordinal


def train_one_fold(train_x: np.ndarray, train_return: np.ndarray, train_label: np.ndarray,
                   seed: int, epochs: int, batch_size: int, device: torch.device) -> SharedMultiTask:
    seed_everything(seed)
    model = SharedMultiTask(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    dataset = TensorDataset(
        torch.from_numpy(train_x.astype(np.float32)),
        torch.from_numpy(train_return.astype(np.float32)),
        torch.from_numpy(train_label.astype(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=(device.type == "cuda"), drop_last=False)
    model.train()
    for _ in range(epochs):
        for x, y_return, y_label in loader:
            if device.type == "cuda":
                x = x.to(device, non_blocking=True); y_return = y_return.to(device, non_blocking=True); y_label = y_label.to(device, non_blocking=True)
            else:
                x = x.to(device); y_return = y_return.to(device); y_label = y_label.to(device)
            optimizer.zero_grad(set_to_none=True)
            mu, logvar, eta, thresholds = model(x)
            loss = loss_fn(mu, logvar, eta, thresholds, y_return, y_label)
            if not torch.isfinite(loss):
                raise RuntimeError("WP24 non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def robust_fit(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train_x, axis=0)
    q25 = np.nanpercentile(train_x, 25, axis=0)
    q75 = np.nanpercentile(train_x, 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    median[~np.isfinite(median)] = 0.0
    return median.astype(np.float64), scale.astype(np.float64)


def robust_transform(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    transformed = (values - median) / scale
    return np.clip(transformed, -8.0, 8.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--rg3-input", required=True, type=Path)
    parser.add_argument("--rg2-input", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve(); rg3_path = args.rg3_input.resolve(); rg2_path = args.rg2_input.resolve(); protocol_path = args.protocol.resolve(); output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    if any(token in str(path).lower() for path in (source, rg3_path, rg2_path, protocol_path, output) for token in BAN):
        raise RuntimeError("prohibited path token")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_TRAINING_OR_METRIC_READ":
        raise RuntimeError("WP24 protocol is not frozen")
    expected = protocol["immutable_inputs"]
    samples_path = source / "data/rg1_4_materialized/samples.csv.gz"
    split_path = source / "governance/rev7_1_freeze/SPLIT_PURGE_EMBARGO_AND_SAMPLE_KEY_CONTRACT.json"
    for label, path, digest in (("samples", samples_path, expected["samples_sha256"]), ("rg3", rg3_path, expected["rg3_features_sha256"]), ("rg2", rg2_path, expected["rg2_state_features_sha256"]), ("split", split_path, expected["split_contract_sha256"])):
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"{label} hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("WP24 requires the shared CUDA environment")
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    device = torch.device("cuda")
    samples = pd.read_csv(samples_path, usecols=["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256", "target_return_h4", "target_valid"], dtype={"fold_id": str, "split_role": str, "stock_code": str, "sample_key_sha256": str})
    samples["trade_date"] = pd.to_datetime(samples["trade_date"], errors="raise").dt.normalize()
    canonical = samples[["trade_date", "stock_code", "target_return_h4", "target_valid"]].drop_duplicates(["trade_date", "stock_code"], keep="first")
    raw = pd.to_numeric(canonical["target_return_h4"], errors="coerce")
    valid = canonical["target_valid"].astype(bool) & raw.notna()
    median = raw.where(valid).groupby(canonical["trade_date"], sort=True).transform("median")
    labels = canonical[["trade_date", "stock_code"]].copy()
    labels["relative_return"] = raw - median
    labels["target_valid"] = valid & median.notna()
    labels["ordinal_target"] = np.select([labels.relative_return < -0.01, labels.relative_return > 0.01], [0, 2], default=1).astype(np.int8)
    # The source row carries the raw validity flag used to build labels.  Drop
    # it before the keyed merge so the canonical derived validity remains the
    # single unambiguous `target_valid` column consumed by the fold masks.
    samples = samples.drop(columns=["target_valid"])
    rg3 = pd.read_csv(rg3_path, dtype={"stock_code": str})
    rg3["trade_date"] = pd.to_datetime(rg3["trade_date"], errors="raise").dt.normalize()
    if rg3.duplicated(["trade_date", "stock_code"]).any():
        raise RuntimeError("RG3 duplicate key")
    rg2 = pd.read_csv(rg2_path, dtype={"fold_id": str, "split_role": str, "stock_code": str, "sample_key_sha256": str})
    rg2["trade_date"] = pd.to_datetime(rg2["trade_date"], errors="raise").dt.normalize()
    if rg2.duplicated("sample_key_sha256").any():
        raise RuntimeError("RG2 duplicate sample key")
    missing = [column for column in FEATURES if column not in rg3.columns and column not in rg2.columns]
    if missing:
        raise RuntimeError(f"feature columns missing: {missing}")
    joined = samples.merge(labels, on=["trade_date", "stock_code"], how="left", validate="many_to_one")
    joined = joined.merge(rg3[["trade_date", "stock_code", *RG3_DAILY, *RG3_STRUCTURAL]], on=["trade_date", "stock_code"], how="left", validate="many_to_one")
    joined = joined.merge(rg2[["sample_key_sha256", *RG2_FEATURES]], on="sample_key_sha256", how="left", validate="one_to_one")
    if joined[FEATURES].isna().any().any() or ~np.isfinite(joined[FEATURES].to_numpy(float)).all():
        raise RuntimeError("WP24 feature join contains missing or non-finite values")
    output.mkdir(parents=True); (output / "predictions_sealed").mkdir(); (output / "checkpoints").mkdir(); (output / "normalization").mkdir(); (output / "run_receipts").mkdir()
    prediction_hashes = {}; checkpoint_hashes = {}; fold_receipts = []
    for index, fold in enumerate(FOLDS, start=1):
        train_mask = (joined.fold_id == fold) & (joined.split_role == "TRAIN") & joined.target_valid.astype(bool) & np.isfinite(joined.relative_return)
        validation_mask = (joined.fold_id == fold) & (joined.split_role == "VALIDATION")
        train = joined.loc[train_mask].copy(); validation = joined.loc[validation_mask].copy()
        train_x_raw = train[FEATURES].to_numpy(np.float64); validation_x_raw = validation[FEATURES].to_numpy(np.float64)
        center, scale = robust_fit(train_x_raw)
        train_x = robust_transform(train_x_raw, center, scale); validation_x = robust_transform(validation_x_raw, center, scale)
        model = train_one_fold(train_x, train.relative_return.to_numpy(np.float64), train.ordinal_target.to_numpy(np.int8), 2026081524 + index, 35, 8192, device)
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(validation_x).to(device)
            mu, logvar, eta, thresholds = model(x)
            probability = probabilities(eta, thresholds).cpu().numpy()
            predicted_return = mu.cpu().numpy(); uncertainty = torch.exp(0.5 * logvar).cpu().numpy()
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(f"WP24 invalid probability output: {fold}")
        predictions = validation[["fold_id", "split_role", "trade_date", "stock_code", "sample_key_sha256"]].copy()
        predictions["candidate_id"] = protocol["model"]["id"]
        predictions["predicted_h4_relative"] = predicted_return
        predictions["predicted_h4_scale"] = uncertainty
        predictions["prob_down"] = probability[:, 0]; predictions["prob_neutral"] = probability[:, 1]; predictions["prob_up"] = probability[:, 2]
        predictions["predicted_ordinal"] = np.argmax(probability, axis=1).astype(np.int8)
        predictions = predictions.sort_values("sample_key_sha256", kind="mergesort")
        prediction_path = output / "predictions_sealed" / f"{fold}_WP24.parquet"; predictions.to_parquet(prediction_path, index=False, engine="pyarrow", compression="zstd")
        norm_path = output / "normalization" / f"{fold}.json"; norm_path.write_text(json.dumps({"fold_id": fold, "features": FEATURES, "median": center.tolist(), "iqr": scale.tolist()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checkpoint_path = output / "checkpoints" / f"{fold}.pt"; torch.save({"candidate_id": protocol["model"]["id"], "fold_id": fold, "feature_names": FEATURES, "state_dict": model.state_dict(), "median": center, "iqr": scale}, checkpoint_path)
        prediction_hashes[str(prediction_path.relative_to(output)).replace("\\", "/")] = sha256(prediction_path); checkpoint_hashes[str(checkpoint_path.relative_to(output)).replace("\\", "/")] = sha256(checkpoint_path)
        fold_receipts.append({"fold_id": fold, "train_rows": int(len(train)), "validation_rows": int(len(validation)), "training_device": torch.cuda.get_device_name(0), "metrics_read": False, "targets_written": False})
        del model, x
        torch.cuda.empty_cache()
    manifest = {"node_id": "WP24_H4_T2_MULTITASK_PREDICTION_SEAL", "status": "SEALED_PENDING_INDEPENDENT_METRIC_READ", "candidate_id": protocol["model"]["id"], "protocol_sha256": sha256(protocol_path), "input_sha256": {"samples": sha256(samples_path), "rg3": sha256(rg3_path), "rg2": sha256(rg2_path), "split": sha256(split_path)}, "prediction_sha256": prediction_hashes, "checkpoint_sha256": checkpoint_hashes, "folds": list(FOLDS), "metrics_read": False, "targets_written": False, "fresh_payloads_opened": False, "production_replacement_allowed": False, "gpu_used": True}
    (output / "PREDICTION_SEAL_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "EXECUTION_RECEIPT.json").write_text(json.dumps({"node_id": "WP24_H4_T2_MULTITASK_GPU_V1", "status": "PASS_PREDICTIONS_SEALED_PENDING_INDEPENDENT_METRIC_READ", "fold_receipts": fold_receipts, "created_at_utc": datetime.now(timezone.utc).isoformat(), "metrics_read": False, "fresh_payloads_opened": False, "production_replacement_allowed": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output_root": str(output), "folds": len(FOLDS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
