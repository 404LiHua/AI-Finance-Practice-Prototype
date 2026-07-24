from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.core import DataBundle, write_json
from stage_c.models import GraphFrequencyModel


class GraphFrequencyExperiment:
    name = "graph_frequency_v1"

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.model: GraphFrequencyModel | None = None
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.device = "cpu"

    def _prepare_scaler(self, data: DataBundle) -> None:
        values = data.panel.loc[data.panel["split"] == "train", data.feature_columns].to_numpy(np.float32)
        self.medians = np.nanmedian(values, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        filled = np.where(np.isfinite(values), values, self.medians)
        self.means = filled.mean(axis=0)
        self.stds = filled.std(axis=0)
        self.stds = np.where(self.stds < 1e-8, 1.0, self.stds)

    def _scaled_panel(self, data: DataBundle) -> pd.DataFrame:
        assert self.medians is not None and self.means is not None and self.stds is not None
        values = data.panel[data.feature_columns].to_numpy(np.float32)
        values = np.where(np.isfinite(values), values, self.medians)
        values = (values - self.means) / self.stds
        result = data.panel[["stock_code", "trade_date"]].copy()
        result["vector"] = list(values.astype(np.float32))
        return result

    def sequences(self, data: DataBundle, split: str) -> tuple[np.ndarray, np.ndarray]:
        scaled = self._scaled_panel(data)
        sequence_length = int(self.config["sequence_length"])
        lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
        for stock_code, frame in scaled.groupby("stock_code", sort=True):
            vectors = frame["vector"].tolist()
            for index, date in enumerate(frame["trade_date"].tolist()):
                sequence = vectors[max(0, index - sequence_length + 1): index + 1]
                if len(sequence) < sequence_length:
                    sequence = [sequence[0]] * (sequence_length - len(sequence)) + sequence
                lookup[(stock_code, date)] = np.stack(sequence)
        samples = data.samples[split]
        x = np.stack([lookup[(row.stock_code, row.trade_date)] for row in samples.itertuples(index=False)])
        y = samples["target_return"].to_numpy(np.float32)
        return x.astype(np.float32), y

    @staticmethod
    def regularization(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        entropy = -(adjacency.clamp_min(1e-8) * adjacency.clamp_min(1e-8).log()).sum(-1).mean()
        smoothness = (adjacency[:, 1:] - adjacency[:, :-1]).abs().mean()
        return entropy, smoothness

    def fit(self, data: DataBundle, output_dir: Path, logger: Any) -> None:
        self._prepare_scaler(data)
        train_x, train_y = self.sequences(data, "train")
        validation_x, validation_y = self.sequences(data, "validation")
        requested = str(self.config.get("device", "auto"))
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"

        model_keys = (
            "hidden_dim", "nhead", "num_temporal_layers", "dim_feedforward", "dropout",
            "top_k", "gumbel_temperature", "keep_self_loops", "graph_mode",
            "use_frequency", "fusion_mode",
            "sampling_mode",
            "use_time_graph",
        )
        model_config = {key: self.config[key] for key in model_keys}
        self.model = GraphFrequencyModel(
            input_dim=train_x.shape[-1],
            sequence_length=train_x.shape[1],
            **model_config,
        ).to(self.device)
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info("Graph-frequency model parameters=%d device=%s", parameter_count, self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        prediction_loss = nn.HuberLoss()
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=int(self.config["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        validation_x_tensor = torch.from_numpy(validation_x).to(self.device)
        validation_y_tensor = torch.from_numpy(validation_y).to(self.device)
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_temperature = float(self.config["gumbel_temperature"])
        stale_epochs = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, int(self.config["epochs"]) + 1):
            temperature = float(self.config["gumbel_temperature"])
            if self.config.get("temperature_schedule", "constant") == "linear":
                start = float(self.config["temperature_start"])
                end = float(self.config["temperature_end"])
                progress = (epoch - 1) / max(int(self.config["epochs"]) - 1, 1)
                temperature = start + (end - start) * progress
            if self.model.graph_learner is not None:
                self.model.graph_learner.set_temperature(temperature)
            self.model.train()
            batch_rows = []
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                details = self.model(batch_x, return_details=True)
                base_loss = prediction_loss(details["prediction"], batch_y)
                if self.config["graph_mode"] == "learned" and (
                    self.config["use_frequency"] or self.config["use_time_graph"]
                ):
                    entropy, smoothness = self.regularization(details["adjacency"])
                else:
                    entropy = smoothness = torch.zeros((), device=self.device)
                loss = (
                    base_loss
                    + float(self.config["lambda_entropy"]) * entropy
                    + float(self.config["lambda_smooth"]) * smoothness
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite Stage C training loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.config["gradient_clip"]),
                )
                optimizer.step()
                batch_rows.append((float(loss.detach().cpu()), float(gradient_norm.detach().cpu())))

            self.model.eval()
            with torch.no_grad():
                validation_details = self.model(validation_x_tensor, return_details=True)
                validation_loss = float(prediction_loss(
                    validation_details["prediction"], validation_y_tensor,
                ).cpu())
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean([item[0] for item in batch_rows])),
                "validation_loss": validation_loss,
                "gradient_norm": float(np.mean([item[1] for item in batch_rows])),
                "temperature": temperature,
            }
            history.append(row)
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                best_temperature = temperature
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % 5 == 0:
                logger.info(
                    "epoch=%d train_loss=%.8f validation_loss=%.8f gradient_norm=%.5f",
                    epoch, row["train_loss"], validation_loss, row["gradient_norm"],
                )
            if stale_epochs >= int(self.config["patience"]):
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.model.graph_learner is not None:
            self.model.graph_learner.set_temperature(best_temperature)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "feature_columns": data.feature_columns,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "seed": self.seed,
        }, output_dir / "model.pt")
        write_json(output_dir / "training_history.json", history)
        write_json(output_dir / "model_metadata.json", {
            "model": self.name,
            "parameter_count": parameter_count,
            "input_shape": ["batch", int(self.config["sequence_length"]), len(data.feature_columns)],
            "node_definition": "weekly positions inside each stock sequence",
            "graph_mode": self.config["graph_mode"],
            "use_frequency": self.config["use_frequency"],
            "use_time_graph": self.config["use_time_graph"],
            "fusion_mode": self.config["fusion_mode"],
            "top_k": self.config["top_k"],
            "sampling_mode": self.config["sampling_mode"],
            "temperature_schedule": self.config.get("temperature_schedule", "constant"),
            "selected_temperature": best_temperature,
            "architecture": "Transformer temporal branch + configurable graph + optional frequency/time-domain propagation + configurable fusion",
        })

    def predict_with_diagnostics(
        self, data: DataBundle, split: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("model has not been fitted")
        x, _ = self.sequences(data, split)
        self.model.eval()
        with torch.no_grad():
            details = self.model(torch.from_numpy(x).to(self.device), return_details=True)
        return (
            details["prediction"].cpu().numpy().astype(float),
            details["adjacency"].cpu().numpy().astype(float),
            details["gate"].cpu().numpy().astype(float),
        )
