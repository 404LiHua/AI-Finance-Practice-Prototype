from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from experiments.core import DataBundle, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Upstream model file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream model module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TorchUpstreamAdapter:
    name = "upstream_adapter"
    source_relative_path = ""

    def __init__(self, config: dict[str, Any], external: dict[str, Any], seed: int) -> None:
        self.config = config
        self.external = external
        self.seed = seed
        self.model: Any = None
        self.device = "cpu"
        self.feature_columns: list[str] = []
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.target_mean = 0.0
        self.target_std = 1.0

    @property
    def source_file(self) -> Path:
        return Path(self.external["root"]) / self.source_relative_path

    def select_features(self, data: DataBundle) -> list[str]:
        raise NotImplementedError

    def build_network(self, input_size: int, sequence_length: int) -> Any:
        raise NotImplementedError

    def model_output(self, values: Any) -> Any:
        return self.model(values).reshape(values.shape[0], -1)[:, 0]

    def scale_target(self) -> bool:
        return False

    def _prepare_scaler(self, data: DataBundle) -> None:
        self.feature_columns = self.select_features(data)
        train_rows = data.panel[data.panel["split"] == "train"][self.feature_columns]
        values = train_rows.to_numpy(dtype=np.float32)
        self.medians = np.nanmedian(values, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        filled = np.where(np.isfinite(values), values, self.medians)
        self.means = filled.mean(axis=0)
        self.stds = filled.std(axis=0)
        self.stds = np.where(self.stds < 1e-8, 1.0, self.stds)
        if self.scale_target():
            target = data.samples["train"]["target_return"].to_numpy(dtype=np.float32)
            self.target_mean = float(np.mean(target))
            self.target_std = float(np.std(target))
            if self.target_std < 1e-8:
                self.target_std = 1.0

    def _scaled_panel(self, data: DataBundle) -> pd.DataFrame:
        assert self.medians is not None and self.means is not None and self.stds is not None
        values = data.panel[self.feature_columns].to_numpy(dtype=np.float32)
        values = np.where(np.isfinite(values), values, self.medians)
        values = (values - self.means) / self.stds
        scaled = data.panel[["stock_code", "trade_date"]].copy()
        scaled["vector"] = list(values.astype(np.float32))
        return scaled

    def _sequences(self, data: DataBundle, split: str) -> tuple[np.ndarray, np.ndarray]:
        scaled = self._scaled_panel(data)
        sequence_length = int(self.config["sequence_length"])
        lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
        for stock_code, frame in scaled.groupby("stock_code", sort=True):
            vectors = frame["vector"].tolist()
            dates = frame["trade_date"].tolist()
            for index, date in enumerate(dates):
                start = max(0, index - sequence_length + 1)
                sequence = vectors[start:index + 1]
                if len(sequence) < sequence_length:
                    sequence = [sequence[0]] * (sequence_length - len(sequence)) + sequence
                lookup[(stock_code, date)] = np.stack(sequence)
        samples = data.samples[split]
        x = np.stack([lookup[(row.stock_code, row.trade_date)] for row in samples.itertuples(index=False)])
        y = samples["target_return"].to_numpy(dtype=np.float32)
        if self.scale_target():
            y = (y - self.target_mean) / self.target_std
        return x.astype(np.float32), y.astype(np.float32)

    def fit(self, data: DataBundle, output_dir: Path, logger: Any) -> None:
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError(f"{self.name} adapter requires PyTorch") from exc

        self._prepare_scaler(data)
        train_x, train_y = self._sequences(data, "train")
        validation_x, validation_y = self._sequences(data, "validation")
        requested = str(self.config.get("device", "auto"))
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"
        self.model = self.build_network(train_x.shape[-1], train_x.shape[1]).to(self.device)
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info("%s parameters=%d device=%s source=%s", self.name, parameter_count, self.device, self.source_file)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        loss_fn = nn.HuberLoss() if self.config.get("loss") == "huber" else nn.MSELoss()
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=int(self.config["batch_size"]), shuffle=True, generator=generator,
        )
        validation_x_tensor = torch.from_numpy(validation_x).to(self.device)
        validation_y_tensor = torch.from_numpy(validation_y).to(self.device)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, int(self.config["epochs"]) + 1):
            self.model.train()
            losses = []
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_output(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["gradient_clip"]))
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.model.eval()
            with torch.no_grad():
                validation_loss = float(loss_fn(self.model_output(validation_x_tensor), validation_y_tensor).cpu())
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            })
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % 10 == 0:
                logger.info(
                    "%s epoch=%d train_loss=%.8f validation_loss=%.8f",
                    self.name, epoch, history[-1]["train_loss"], validation_loss,
                )
            if stale_epochs >= int(self.config["patience"]):
                logger.info("%s early stopping at epoch %d", self.name, epoch)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        source_sha256 = _sha256(self.source_file)
        torch.save({
            "state_dict": self.model.state_dict(),
            "adapter_config": self.config,
            "feature_columns": self.feature_columns,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "source_file": str(self.source_file),
            "source_sha256": source_sha256,
            "seed": self.seed,
        }, output_dir / "model.pt")
        write_json(output_dir / "training_history.json", history)
        write_json(output_dir / "model_metadata.json", {
            "model": self.name,
            "adapter": "unified Stage B split/sequence/evaluator",
            "parameter_count": parameter_count,
            "feature_columns": self.feature_columns,
            "sequence_length": int(self.config["sequence_length"]),
            "source_file": str(self.source_file),
            "source_sha256": source_sha256,
            "upstream_entrypoint": self.external.get("entrypoint"),
            "sample_counts": {split: len(frame) for split, frame in data.samples.items()},
        })

    def predict(self, data: DataBundle, split: str) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        x, _ = self._sequences(data, split)
        self.model.eval()
        predictions = []
        batch_size = int(self.config["batch_size"])
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = torch.from_numpy(x[start:start + batch_size]).to(self.device)
                predictions.append(self.model_output(batch).cpu().numpy())
        result = np.concatenate(predictions).astype(float)
        if self.scale_target():
            result = result * self.target_std + self.target_mean
        return result


class FreTSAdapter(TorchUpstreamAdapter):
    name = "frets"
    source_relative_path = "models/FreTS.py"

    def select_features(self, data: DataBundle) -> list[str]:
        if "return_1w" not in data.panel.columns:
            raise KeyError("FreTS adapter requires return_1w")
        return ["return_1w"]

    def scale_target(self) -> bool:
        return True

    def build_network(self, input_size: int, sequence_length: int) -> Any:
        module = _load_module(f"qrg_frets_{self.seed}", self.source_file)
        configs = SimpleNamespace(
            pred_len=1,
            enc_in=input_size,
            seq_len=sequence_length,
            channel_independence=str(self.config.get("channel_independence", "0")),
        )
        return module.Model(configs)


class TimeGNNAdapter(TorchUpstreamAdapter):
    name = "timegnn"
    source_relative_path = "models/TimeGNN.py"

    def select_features(self, data: DataBundle) -> list[str]:
        return list(data.feature_columns)

    def build_network(self, input_size: int, sequence_length: int) -> Any:
        import torch

        module = _load_module(f"qrg_timegnn_{self.seed}", self.source_file)
        return module.TimeGNN(
            loss=torch.nn.HuberLoss(),
            input_dim=input_size,
            hidden_dim=int(self.config["hidden_dim"]),
            output_dim=1,
            seq_len=sequence_length,
            batch_size=int(self.config["batch_size"]),
            aggregate=str(self.config.get("aggregate", "last")),
            keep_self_loops=bool(self.config.get("keep_self_loops", False)),
            enforce_consecutive=bool(self.config.get("enforce_consecutive", False)),
            block_size=int(self.config["block_size"]),
        )
