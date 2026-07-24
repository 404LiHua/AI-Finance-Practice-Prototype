from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.core import DataBundle


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frets_source(path: Path, seed: int) -> Any:
    spec = importlib.util.spec_from_file_location(f"stage_d_frozen_frets_{seed}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen FreTS source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenFreTSCheckpoint:
    def __init__(
        self, checkpoint_path: Path, checkpoint_sha256: str,
        source_path: Path, source_sha256: str, expected_seed: int, device: str = "cpu",
    ) -> None:
        self.checkpoint_path = checkpoint_path.resolve()
        self.source_path = source_path.resolve()
        if sha256_file(self.checkpoint_path) != checkpoint_sha256:
            raise ValueError(f"checkpoint SHA-256 mismatch: {self.checkpoint_path}")
        if sha256_file(self.source_path) != source_sha256:
            raise ValueError(f"frozen FreTS source SHA-256 mismatch: {self.source_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        required = {
            "state_dict", "adapter_config", "feature_columns", "medians", "means", "stds",
            "target_mean", "target_std", "source_sha256", "seed",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(f"frozen FreTS checkpoint missing fields: {sorted(missing)}")
        self.seed = int(checkpoint["seed"])
        if self.seed != int(expected_seed):
            raise ValueError("frozen checkpoint seed mismatch")
        if checkpoint["source_sha256"] != source_sha256:
            raise ValueError("checkpoint upstream source hash differs from frozen source")
        self.config = dict(checkpoint["adapter_config"])
        self.feature_columns = list(checkpoint["feature_columns"])
        if self.feature_columns != ["return_1w"] or int(self.config["sequence_length"]) != 4:
            raise ValueError("frozen FreTS feature order or sequence length mismatch")
        self.medians = np.asarray(checkpoint["medians"], dtype=np.float32)
        self.means = np.asarray(checkpoint["means"], dtype=np.float32)
        self.stds = np.asarray(checkpoint["stds"], dtype=np.float32)
        self.target_mean = float(checkpoint["target_mean"])
        self.target_std = float(checkpoint["target_std"])
        self.device = torch.device(device)
        module = _load_frets_source(self.source_path, self.seed)
        args = SimpleNamespace(
            pred_len=1, enc_in=1, seq_len=4,
            channel_independence=str(self.config.get("channel_independence", "0")),
        )
        self.model = module.Model(args).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    def _sequences(self, data: DataBundle, split: str) -> np.ndarray:
        if "return_1w" not in data.panel.columns:
            raise KeyError("frozen FreTS requires return_1w")
        values = data.panel[["return_1w"]].to_numpy(dtype=np.float32)
        values = np.where(np.isfinite(values), values, self.medians)
        values = (values - self.means) / self.stds
        scaled = data.panel[["stock_code", "trade_date"]].copy()
        scaled["vector"] = list(values.astype(np.float32))
        lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
        for stock_code, frame in scaled.groupby("stock_code", sort=True):
            vectors = frame["vector"].tolist()
            for index, date in enumerate(frame["trade_date"].tolist()):
                sequence = vectors[max(0, index - 3): index + 1]
                if len(sequence) < 4:
                    sequence = [sequence[0]] * (4 - len(sequence)) + sequence
                lookup[(stock_code, date)] = np.stack(sequence)
        samples = data.samples[split]
        if samples.empty:
            raise ValueError(f"split has no eligible samples: {split}")
        return np.stack([
            lookup[(row.stock_code, row.trade_date)]
            for row in samples.itertuples(index=False)
        ]).astype(np.float32)

    def predict(self, data: DataBundle, split: str) -> np.ndarray:
        values = self._sequences(data, split)
        with torch.no_grad():
            raw = self.model(torch.from_numpy(values).to(self.device)).reshape(len(values), -1)[:, 0]
        prediction = raw.cpu().numpy().astype(float) * self.target_std + self.target_mean
        if not np.isfinite(prediction).all():
            raise FloatingPointError("frozen FreTS produced non-finite predictions")
        return prediction


class LoadedStageDFrozenCandidate:
    def __init__(self, manifest_path: Path, repo_root: Path, device: str = "cpu") -> None:
        self.manifest_path = manifest_path.resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest["candidate_model"] != "frets_return_l4__fixed_shrink_a075":
            raise ValueError("unexpected frozen Stage D candidate")
        self.alpha = float(manifest["shrinkage_alpha"])
        if self.alpha != 0.75 or manifest["aggregation"] != "arithmetic_mean":
            raise ValueError("frozen shrinkage or aggregation mismatch")
        source_path = (repo_root / manifest["upstream_source"]["path"]).resolve()
        source_sha = manifest["upstream_source"]["sha256"]
        self.components = []
        for item in manifest["checkpoints"]:
            checkpoint_path = (repo_root / item["path"]).resolve()
            self.components.append(FrozenFreTSCheckpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=item["sha256"],
                source_path=source_path,
                source_sha256=source_sha,
                expected_seed=int(item["seed"]),
                device=device,
            ))
        if [item.seed for item in self.components] != [20260723, 20260724, 20260725]:
            raise ValueError("frozen Stage D seed set mismatch")

    def predict(self, data: DataBundle, split: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        per_seed = {component.seed: self.alpha * component.predict(data, split) for component in self.components}
        aggregate = np.mean(np.stack(list(per_seed.values()), axis=0), axis=0)
        if not np.isfinite(aggregate).all():
            raise FloatingPointError("frozen Stage D aggregate produced non-finite predictions")
        return aggregate, per_seed

    @staticmethod
    def naive(samples: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(samples), dtype=float)
