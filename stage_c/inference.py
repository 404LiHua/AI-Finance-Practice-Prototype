from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.core import DataBundle
from stage_c.models import GraphFrequencyModel


MODEL_KEYS = (
    "hidden_dim", "nhead", "num_temporal_layers", "dim_feedforward", "dropout",
    "top_k", "gumbel_temperature", "keep_self_loops", "graph_mode",
    "use_frequency", "fusion_mode", "sampling_mode", "use_time_graph",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


class FrozenSequenceBuilder:
    """Rebuild sequences using scaler state stored in a fitted checkpoint."""

    def __init__(
        self,
        feature_columns: list[str],
        medians: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        sequence_length: int,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.medians = np.asarray(medians, dtype=np.float32)
        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)
        self.sequence_length = int(sequence_length)
        expected = len(self.feature_columns)
        if any(array.shape != (expected,) for array in (self.medians, self.means, self.stds)):
            raise ValueError("checkpoint scaler shape does not match feature columns")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")

    def build(self, data: DataBundle, split: str) -> np.ndarray:
        if data.feature_columns != self.feature_columns:
            raise ValueError(
                "feature order mismatch: "
                f"checkpoint={self.feature_columns}, data={data.feature_columns}"
            )
        values = data.panel[self.feature_columns].to_numpy(dtype=np.float32)
        values = np.where(np.isfinite(values), values, self.medians)
        values = (values - self.means) / self.stds
        scaled = data.panel[["stock_code", "trade_date"]].copy()
        scaled["vector"] = list(values.astype(np.float32))
        lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
        for stock_code, frame in scaled.groupby("stock_code", sort=True):
            vectors = frame["vector"].tolist()
            for index, date in enumerate(frame["trade_date"].tolist()):
                sequence = vectors[max(0, index - self.sequence_length + 1): index + 1]
                if len(sequence) < self.sequence_length:
                    sequence = [sequence[0]] * (self.sequence_length - len(sequence)) + sequence
                lookup[(stock_code, date)] = np.stack(sequence)
        samples = data.samples[split]
        if samples.empty:
            raise ValueError(f"split has no eligible samples: {split}")
        return np.stack([
            lookup[(row.stock_code, row.trade_date)]
            for row in samples.itertuples(index=False)
        ]).astype(np.float32)


class LoadedStageCComponent:
    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path.resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"component checkpoint does not exist: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        required = {"state_dict", "config", "feature_columns", "medians", "means", "stds", "seed"}
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(f"checkpoint is missing fields: {sorted(missing)}")
        self.config = dict(checkpoint["config"])
        self.seed = int(checkpoint["seed"])
        self.feature_columns = list(checkpoint["feature_columns"])
        self.device = torch.device(device)
        model_config = {
            "hidden_dim": int(self.config.get("hidden_dim", 32)),
            "nhead": int(self.config.get("nhead", 4)),
            "num_temporal_layers": int(self.config.get("num_temporal_layers", 1)),
            "dim_feedforward": int(self.config.get("dim_feedforward", 64)),
            "dropout": float(self.config.get("dropout", 0.1)),
            "top_k": int(self.config.get("top_k", 2)),
            "gumbel_temperature": float(self.config.get("gumbel_temperature", 0.8)),
            "keep_self_loops": bool(self.config.get("keep_self_loops", True)),
            "graph_mode": str(self.config.get("graph_mode", "learned")),
            "use_frequency": bool(self.config.get("use_frequency", True)),
            "fusion_mode": str(self.config.get("fusion_mode", "gated")),
            "sampling_mode": str(self.config.get("sampling_mode", "gumbel")),
            "use_time_graph": bool(self.config.get("use_time_graph", False)),
        }
        self.model = GraphFrequencyModel(
            input_dim=len(self.feature_columns),
            sequence_length=int(self.config["sequence_length"]),
            **model_config,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.sequence_builder = FrozenSequenceBuilder(
            feature_columns=self.feature_columns,
            medians=checkpoint["medians"],
            means=checkpoint["means"],
            stds=checkpoint["stds"],
            sequence_length=int(self.config["sequence_length"]),
        )

    def predict(self, data: DataBundle, split: str) -> np.ndarray:
        values = self.sequence_builder.build(data, split)
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(values).to(self.device))
        result = prediction.cpu().numpy().astype(float)
        if not np.isfinite(result).all():
            raise FloatingPointError(f"non-finite predictions from {self.checkpoint_path}")
        return result

    def provenance(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "sha256": sha256_file(self.checkpoint_path),
            "seed": self.seed,
            "feature_columns": self.feature_columns,
            "sequence_length": int(self.config["sequence_length"]),
            "graph_mode": self.config.get("graph_mode"),
            "use_frequency": self.config.get("use_frequency"),
            "use_time_graph": self.config.get("use_time_graph", False),
        }


@dataclass
class EnsembleComponent:
    name: str
    weight: float
    model: LoadedStageCComponent


class LoadedFixedEnsemble:
    def __init__(self, manifest_path: Path, repo_root: Path, device: str = "cpu") -> None:
        self.manifest_path = manifest_path.resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        components = manifest.get("components", [])
        if len(components) != 2:
            raise ValueError("recommended v2 manifest must contain exactly two components")
        self.components = []
        for index, item in enumerate(components):
            checkpoint = resolve_artifact_path(str(item["model"]), repo_root)
            self.components.append(EnsembleComponent(
                name=str(item.get("component", f"component_{index}")),
                weight=float(item["weight"]),
                model=LoadedStageCComponent(checkpoint, device=device),
            ))
        total_weight = sum(component.weight for component in self.components)
        if any(component.weight < 0 for component in self.components) or abs(total_weight - 1.0) > 1e-8:
            raise ValueError("ensemble weights must be non-negative and sum to one")
        first = self.components[0].model
        for component in self.components[1:]:
            if component.model.feature_columns != first.feature_columns:
                raise ValueError("ensemble checkpoints use different feature columns")
            if component.model.seed != first.seed:
                raise ValueError("ensemble checkpoints must use the same frozen seed")
        self.seed = first.seed
        self.feature_columns = first.feature_columns

    def predict(self, data: DataBundle, split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        component_predictions = {
            component.name: component.model.predict(data, split)
            for component in self.components
        }
        lengths = {len(values) for values in component_predictions.values()}
        if len(lengths) != 1:
            raise ValueError("component prediction lengths do not match")
        ensemble = sum(
            component.weight * component_predictions[component.name]
            for component in self.components
        )
        return ensemble, component_predictions

    def provenance(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "manifest_sha256": sha256_file(self.manifest_path),
            "seed": self.seed,
            "components": [
                {"name": component.name, "weight": component.weight, **component.model.provenance()}
                for component in self.components
            ],
        }

