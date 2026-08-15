from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from stage_e.e5.interface import E5FoldView
from stage_e.e5.low_cost import flattened_samples
from stage_e.run_e3_training_checks import set_seed


class LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        layers = int(config["num_layers"])
        self.lstm = nn.LSTM(
            input_size, hidden, num_layers=layers, batch_first=True,
            dropout=float(config["dropout"]) if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(values)
        return self.head(output[:, -1]).squeeze(-1)


class TemporalBlock(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden, hidden, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(1, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.dropout(torch.relu(self.norm(self.conv(values))))


class TCNNetwork(nn.Module):
    def __init__(self, input_size: int, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_channels"])
        self.input_projection = nn.Conv1d(input_size, hidden, 1)
        self.blocks = nn.ModuleList([
            TemporalBlock(hidden, int(config["kernel_size"]), int(dilation), float(config["dropout"]))
            for dilation in config["dilations"]
        ])
        self.head = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(values.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden[:, :, -1]).squeeze(-1)


class StableTimeGNNNetwork(nn.Module):
    """Time-GNN-style temporal graph with deterministic Top-k edges and dense propagation."""

    def __init__(self, input_size: int, sequence_length: int, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_dim"])
        self.sequence_length = sequence_length
        self.top_k = min(int(config["top_k"]), max(sequence_length - 1, 1))
        self.temperature = float(config["temperature"])
        self.branch1 = nn.Conv1d(input_size, hidden, 1, padding=0)
        self.branch3 = nn.Conv1d(input_size, hidden, 3, padding=1)
        self.branch5 = nn.Conv1d(input_size, hidden, 5, padding=2)
        self.fuse = nn.Linear(hidden * 3, hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.graph_layers = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(int(config["graph_layers"]))])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(int(config["graph_layers"]))])
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def adjacency(self, hidden: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.query(hidden), self.key(hidden).transpose(1, 2)) / math.sqrt(hidden.shape[-1])
        diagonal = torch.eye(self.sequence_length, dtype=torch.bool, device=hidden.device).unsqueeze(0)
        scores = scores.masked_fill(diagonal, float("-inf"))
        top_values, top_indices = torch.topk(scores, self.top_k, dim=-1, largest=True, sorted=True)
        weights = torch.softmax(top_values / self.temperature, dim=-1)
        adjacency = torch.zeros_like(scores).scatter(-1, top_indices, weights)
        return adjacency

    def forward(self, values: torch.Tensor, return_adjacency: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        channels = values.transpose(1, 2)
        hidden = torch.cat([self.branch1(channels), self.branch3(channels), self.branch5(channels)], dim=1)
        hidden = torch.relu(self.fuse(hidden.transpose(1, 2)))
        adjacency = self.adjacency(hidden)
        for layer, norm in zip(self.graph_layers, self.norms):
            propagated = torch.matmul(adjacency, hidden)
            hidden = norm(hidden + torch.relu(layer(torch.cat([hidden, propagated], dim=-1))))
        prediction = self.head(hidden[:, -1]).squeeze(-1)
        return (prediction, adjacency) if return_adjacency else prediction


class FixedIndustryGraphWaveNet(nn.Module):
    """Graph-WaveNet-class baseline over real stock nodes and a fixed industry graph."""

    def __init__(self, input_size: int, adjacency: np.ndarray, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_channels"])
        self.register_buffer("adjacency", torch.from_numpy(adjacency.astype(np.float32)))
        self.input_projection = nn.Conv1d(input_size, hidden, 1)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        for dilation in config["dilations"]:
            padding = int(dilation) * (int(config["kernel_size"]) - 1) // 2
            self.filter_convs.append(nn.Conv1d(hidden, hidden, int(config["kernel_size"]), padding=padding, dilation=int(dilation)))
            self.gate_convs.append(nn.Conv1d(hidden, hidden, int(config["kernel_size"]), padding=padding, dilation=int(dilation)))
        graph_order = int(config["graph_order"])
        self.graph_order = graph_order
        self.graph_projection = nn.Linear(hidden * (graph_order + 1), hidden)
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.head = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, stocks, features = values.shape
        temporal = values.permute(0, 2, 3, 1).reshape(batch * stocks, features, length)
        hidden = self.input_projection(temporal)
        for filter_conv, gate_conv in zip(self.filter_convs, self.gate_convs):
            hidden = hidden + self.dropout(torch.tanh(filter_conv(hidden)) * torch.sigmoid(gate_conv(hidden)))
        node_hidden = hidden[:, :, -1].reshape(batch, stocks, -1)
        propagated = [node_hidden]
        current = node_hidden
        for _ in range(self.graph_order):
            current = torch.einsum("nm,bmh->bnh", self.adjacency, current)
            propagated.append(current)
        fused = torch.relu(self.graph_projection(torch.cat(propagated, dim=-1)))
        return self.head(fused).squeeze(-1)


def fixed_industry_adjacency(stock_codes: np.ndarray, universe: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    ordered = pd.DataFrame({"stock_code": stock_codes.astype(str)}).merge(
        universe[["stock_code", "industry_group"]].assign(stock_code=lambda frame: frame["stock_code"].astype(str)),
        on="stock_code", how="left", validate="one_to_one",
    )
    industries = ordered["industry_group"].fillna("UNKNOWN").astype(str).tolist()
    adjacency = np.equal.outer(industries, industries).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)
    return adjacency, industries


def build_model(
    family: str, input_size: int, sequence_length: int, parameters: dict[str, Any], adjacency: np.ndarray | None,
) -> nn.Module:
    if family == "lstm":
        return LSTMNetwork(input_size, parameters)
    if family == "tcn":
        return TCNNetwork(input_size, parameters)
    if family == "timegnn_stable":
        return StableTimeGNNNetwork(input_size, sequence_length, parameters)
    if family == "stock_node_gwnet_fixed_industry":
        if adjacency is None:
            raise ValueError("stock-node graph baseline requires frozen adjacency")
        return FixedIndustryGraphWaveNet(input_size, adjacency, parameters)
    raise ValueError(f"unsupported E-5.3 family: {family}")


def _masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return nn.functional.huber_loss(prediction[mask], target[mask])


def train_model(
    family: str, view: E5FoldView, parameters: dict[str, Any], seed: int,
    adjacency: np.ndarray | None, checkpoint: Path, log_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed)
    sequence_length = int(parameters["sequence_length"])
    input_size = int(view.numeric_values.shape[-1])
    model = build_model(family, input_size, sequence_length, parameters, adjacency)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(parameters["learning_rate"]), weight_decay=float(parameters["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    target_mean, target_std = float(view.target_mean_train), float(view.target_std_train)
    graph_family = family == "stock_node_gwnet_fixed_industry"
    if graph_family:
        train_indices = view.split_indices("train")
        validation_indices = view.split_indices("validation")
        train_x = view.numeric_values[train_indices, -sequence_length:].astype(np.float32)
        train_y = ((view.target_raw[train_indices] - target_mean) / target_std).astype(np.float32)
        train_mask = view.sample_mask[train_indices].astype(bool)
        validation_x = view.numeric_values[validation_indices, -sequence_length:].astype(np.float32)
        validation_y = ((view.target_raw[validation_indices] - target_mean) / target_std).astype(np.float32)
        validation_mask = view.sample_mask[validation_indices].astype(bool)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_mask),
        )
    else:
        features = list(range(input_size))
        train_x, train_y_raw, _ = flattened_samples(view, "train", sequence_length, features, True)
        validation_x, validation_y_raw, validation_mask = flattened_samples(view, "validation", sequence_length, features, False)
        train_y = ((train_y_raw - target_mean) / target_std).astype(np.float32)
        validation_y = ((validation_y_raw - target_mean) / target_std).astype(np.float32)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(parameters["batch_size"]), shuffle=True, generator=generator,
    )
    validation_x_tensor = torch.from_numpy(validation_x.astype(np.float32))
    validation_y_tensor = torch.from_numpy(validation_y.astype(np.float32))
    validation_mask_tensor = torch.from_numpy(validation_mask.astype(bool))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(parameters["epochs"]) + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            if graph_family:
                batch_x, batch_y, batch_mask = batch
                loss = _masked_huber(model(batch_x), batch_y, batch_mask)
            else:
                batch_x, batch_y = batch
                loss = nn.functional.huber_loss(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {family} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(parameters["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x_tensor)
            validation_loss = float(_masked_huber(validation_prediction, validation_y_tensor, validation_mask_tensor))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(parameters["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"{family} did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    inference_started = time.perf_counter()
    with torch.no_grad():
        scaled_prediction = model(validation_x_tensor).numpy()
    inference_seconds = time.perf_counter() - inference_started
    prediction = scaled_prediction.reshape(-1) * target_std + target_mean
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "family": family, "state_dict": model.state_dict(), "parameters": parameters,
        "input_size": input_size, "target_mean": target_mean, "target_std": target_std,
        "adjacency": adjacency,
    }, checkpoint)
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    detail = {
        "training_seconds": time.perf_counter() - started, "inference_seconds": inference_seconds,
        "epochs_completed": len(history), "best_validation_loss": best_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "real_stock_node_graph": graph_family,
    }
    return prediction.astype(float), detail


def load_predict(checkpoint: Path, view: E5FoldView) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    family = str(payload["family"])
    parameters = dict(payload["parameters"])
    sequence_length = int(parameters["sequence_length"])
    adjacency = payload.get("adjacency")
    model = build_model(family, int(payload["input_size"]), sequence_length, parameters, adjacency)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if family == "stock_node_gwnet_fixed_industry":
        indices = view.split_indices("validation")
        x = view.numeric_values[indices, -sequence_length:].astype(np.float32)
    else:
        x, _, _ = flattened_samples(
            view, "validation", sequence_length, list(range(view.numeric_values.shape[-1])), False,
        )
        x = x.astype(np.float32)
    with torch.no_grad():
        scaled = model(torch.from_numpy(x)).numpy().reshape(-1)
    return scaled.astype(float) * float(payload["target_std"]) + float(payload["target_mean"])
