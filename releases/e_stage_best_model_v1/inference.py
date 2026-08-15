"""Standalone three-seed inference for the frozen Stage-E best model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class FixedIndustryGraphWaveNet(nn.Module):
    def __init__(self, input_size: int, adjacency: np.ndarray, config: dict) -> None:
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
        self.graph_order = int(config["graph_order"])
        self.graph_projection = nn.Linear(hidden * (self.graph_order + 1), hidden)
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.head = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, stocks, features = values.shape
        hidden = self.input_projection(values.permute(0, 2, 3, 1).reshape(batch * stocks, features, length))
        for filter_conv, gate_conv in zip(self.filter_convs, self.gate_convs):
            hidden = hidden + self.dropout(torch.tanh(filter_conv(hidden)) * torch.sigmoid(gate_conv(hidden)))
        node_hidden = hidden[:, :, -1].reshape(batch, stocks, -1)
        propagated = [node_hidden]
        current = node_hidden
        for _ in range(self.graph_order):
            current = torch.einsum("nm,bmh->bnh", self.adjacency, current)
            propagated.append(current)
        return self.head(torch.relu(self.graph_projection(torch.cat(propagated, dim=-1)))).squeeze(-1)


def load_one(path: Path) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["family"] != "stock_node_gwnet_fixed_industry":
        raise ValueError(f"unexpected checkpoint family: {payload['family']}")
    model = FixedIndustryGraphWaveNet(int(payload["input_size"]), payload["adjacency"], payload["parameters"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict(values: np.ndarray, checkpoint_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4 or values.shape[1:] != (8, 100, 6):
        raise ValueError(f"expected [batch,8,100,6], received {values.shape}")
    tensor = torch.from_numpy(values)
    seed_predictions = []
    with torch.no_grad():
        for path in checkpoint_paths:
            model, payload = load_one(path)
            scaled = model(tensor).numpy()
            seed_predictions.append(scaled * float(payload["target_std"]) + float(payload["target_mean"]))
    stacked = np.stack(seed_predictions, axis=0)
    return stacked.mean(axis=0), stacked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="NPZ containing values and optional stock_code")
    parser.add_argument("--fold", choices=["E_RO_01", "E_RO_02", "E_RO_03"], default="E_RO_03")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = np.load(args.input)
    expected_order = json.loads((root / "stock_order.json").read_text(encoding="utf-8"))["stock_codes"]
    if "stock_code" in source and source["stock_code"].astype(str).tolist() != expected_order:
        raise ValueError("input stock order does not match frozen stock_order.json")
    checkpoints = [root / "checkpoints" / args.fold / f"seed_{seed}.pt" for seed in (20260723, 20260724, 20260725)]
    prediction, seed_prediction = predict(source["values"], checkpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, prediction=prediction, seed_prediction=seed_prediction, stock_code=np.asarray(expected_order))
    print(json.dumps({"status": "PASS", "fold": args.fold, "batch": len(prediction), "stocks": prediction.shape[1], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
