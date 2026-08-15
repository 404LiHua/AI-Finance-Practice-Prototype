from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.bounded_ablations import PRICE_FEATURES, minimalist_feature_view
from experiments.core import DataBundle
from stage_d.custody import DataCustodyGuard
from stage_d.rolling_origin import _row_set_sha256


BASE_MODELS = (
    "naive",
    "frets_return_l4",
    "minimalist_price_only_l8",
    "temporal_only_control",
    "fixed_temporal_graph_control",
)


def load_locked_config(path: Path, repo_root: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_LOCKED":
        raise ValueError("D-2 config must be preregistered and locked")
    if tuple(config.get("base_models", ())) != BASE_MODELS:
        raise ValueError("D-2 base model list differs from the preregistration")
    if config["shrinkage"].get("post_result_candidate_additions_allowed") is not False:
        raise ValueError("post-result candidate additions must remain disabled")
    for key in ("protocol_path", "custody_config_path", "data_root", "output_root", "graph_base_config_path"):
        value = Path(config[key])
        config[key] = str((repo_root / value).resolve() if not value.is_absolute() else value.resolve())
    return config


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    protocol = json.loads(Path(config["protocol_path"]).read_text(encoding="utf-8"))
    if protocol["protocol_id"] != config["protocol_id"]:
        raise ValueError("registered protocol id mismatch")
    if protocol["protocol_sha256"] != config["protocol_sha256"]:
        raise ValueError("registered protocol SHA-256 mismatch")
    payload = json.dumps(
        protocol["folds"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != config["protocol_sha256"]:
        raise ValueError("fold definitions no longer match the registered protocol SHA-256")
    if protocol["fold_count"] != 3 or len(protocol["folds"]) != 3:
        raise ValueError("D-2 requires the registered three-fold protocol")
    return protocol


def shrinkage_model_name(base_model: str, alpha: float) -> str:
    return f"{base_model}__fixed_shrink_a{int(round(alpha * 100)):03d}"


def registered_models(config: dict[str, Any]) -> list[str]:
    models = list(BASE_MODELS)
    for base_model in config["shrinkage"]["base_models"]:
        for alpha in config["shrinkage"]["alphas"]:
            models.append(shrinkage_model_name(str(base_model), float(alpha)))
    return models


def apply_fixed_shrinkage(prediction: Any, alpha: float) -> Any:
    allowed = (0.25, 0.5, 0.75)
    if float(alpha) not in allowed:
        raise ValueError(f"unregistered shrinkage alpha: {alpha}")
    return float(alpha) * prediction


def build_fold_bundle(
    config: dict[str, Any], protocol: dict[str, Any], fold_id: str, repo_root: Path,
) -> tuple[DataBundle, dict[str, Any]]:
    fold = next((item for item in protocol["folds"] if item["fold_id"] == fold_id), None)
    if fold is None:
        raise KeyError(f"unknown registered fold: {fold_id}")
    guard = DataCustodyGuard.from_config(Path(config["custody_config_path"]), repo_root)
    data_root = Path(config["data_root"])
    guard.assert_paths_allowed(
        [data_root / "panel.csv.gz", data_root / "text_features.csv.gz"],
        purpose="D-2 rolling-origin development",
    )
    loaded = DataBundle.load(data_root)
    guard.assert_development_frame(loaded.panel)
    panel = loaded.panel.loc[
        loaded.panel["trade_date"].le(pd.Timestamp(fold["validation_end"]))
    ].copy()
    panel["split"] = "context"
    eligible = panel["target_return"].notna() & panel["target_date"].notna()
    if "history_weeks_available" in panel:
        eligible &= panel["history_weeks_available"].ge(int(config["lookback_weeks"]))
    if "cross_section_eligible" in panel:
        eligible &= panel["cross_section_eligible"].fillna(False)
    train_mask = (
        eligible
        & panel["trade_date"].le(pd.Timestamp(fold["train_end"]))
        & panel["target_date"].le(pd.Timestamp(fold["train_end"]))
    )
    validation_mask = (
        eligible
        & panel["trade_date"].between(fold["validation_start"], fold["validation_end"])
        & panel["target_date"].le(pd.Timestamp(fold["validation_end"]))
    )
    panel.loc[train_mask, "split"] = "train"
    panel.loc[validation_mask, "split"] = "validation"
    samples = {
        "train": panel.loc[train_mask].copy().reset_index(drop=True),
        "validation": panel.loc[validation_mask].copy().reset_index(drop=True),
    }
    observed = {
        "train_samples": len(samples["train"]),
        "validation_samples": len(samples["validation"]),
        "train_row_set_sha256": _row_set_sha256(samples["train"]),
        "validation_row_set_sha256": _row_set_sha256(samples["validation"]),
    }
    for key, value in observed.items():
        if value != fold[key]:
            raise RuntimeError(f"{fold_id} {key} differs from registered protocol: {value} != {fold[key]}")
    if panel["stock_code"].nunique() != int(config["stock_count"]):
        raise RuntimeError("D-2 input is not the locked 30-stock panel")
    guard.assert_development_frame(panel)
    return DataBundle(panel=panel, samples=samples, feature_columns=loaded.feature_columns), observed


def price_only_bundle(data: DataBundle) -> DataBundle:
    return minimalist_feature_view(data, "price_only")


def graph_price_bundle(data: DataBundle) -> DataBundle:
    missing = [column for column in PRICE_FEATURES if column not in data.panel.columns]
    if missing:
        raise KeyError(f"Stage D graph controls require missing price features: {missing}")
    return DataBundle(panel=data.panel, samples=data.samples, feature_columns=PRICE_FEATURES.copy())
