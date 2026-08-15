from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn

from stage_e.e5.interface import E5FoldView
from stage_e.hashing import sha256_file
from stage_e.run_e3_training_checks import set_seed


def flattened_samples(
    view: E5FoldView, split_name: str, sequence_length: int, feature_indices: list[int], valid_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = view.split_indices(split_name)
    values = view.numeric_values[indices, -sequence_length:, :, :][:, :, :, feature_indices]
    x = values.transpose(0, 2, 1, 3).reshape(-1, sequence_length, len(feature_indices))
    y_raw = view.target_raw[indices].reshape(-1)
    mask = view.sample_mask[indices].reshape(-1)
    if valid_only:
        return x[mask], y_raw[mask], mask
    return x, y_raw, mask


class MinimalistNetwork(nn.Module):
    def __init__(self, input_size: int, sequence_length: int, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=int(config["nhead"]), dim_feedforward=int(config["dim_feedforward"]),
            dropout=float(config["dropout"]), activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config["num_layers"]), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(values) + self.position[:, : values.shape[1]]
        return self.head(self.norm(self.encoder(encoded)[:, -1])).squeeze(-1)


def load_frets_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen FreTS source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_torch_model(
    family: str, view: E5FoldView, parameters: dict[str, Any], seed: int, source_path: Path | None,
    checkpoint: Path, log_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed)
    feature_indices = [0] if family == "frets" else list(range(view.numeric_values.shape[-1]))
    sequence_length = int(parameters["sequence_length"])
    train_x, train_y_raw, _ = flattened_samples(view, "train", sequence_length, feature_indices, True)
    validation_x, validation_y_raw, validation_mask = flattened_samples(view, "validation", sequence_length, feature_indices, False)
    target_mean, target_std = view.target_mean_train, view.target_std_train
    train_y = ((train_y_raw - target_mean) / target_std).astype(np.float32)
    validation_y = ((validation_y_raw - target_mean) / target_std).astype(np.float32)
    if family == "frets":
        assert source_path is not None
        module = load_frets_module(source_path, f"stage_e_e5_frets_{seed}_{view.fold_id}")
        model = module.Model(SimpleNamespace(
            pred_len=1, enc_in=1, seq_len=sequence_length,
            channel_independence=str(parameters["channel_independence"]),
        ))
    else:
        model = MinimalistNetwork(train_x.shape[-1], sequence_length, parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(parameters["learning_rate"]), weight_decay=float(parameters["weight_decay"]))
    loss_function = nn.HuberLoss()
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(parameters["batch_size"]), shuffle=True, generator=generator)
    val_x_tensor = torch.from_numpy(validation_x.astype(np.float32))
    val_y_tensor = torch.from_numpy(validation_y)
    val_mask_tensor = torch.from_numpy(validation_mask.astype(bool))
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(parameters["epochs"]) + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            if family == "frets":
                output = output[:, 0, 0]
            loss = loss_function(output, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {family} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(parameters["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_output = model(val_x_tensor)
            if family == "frets":
                val_output = val_output[:, 0, 0]
            val_loss = float(loss_function(val_output[val_mask_tensor], val_y_tensor[val_mask_tensor]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
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
    with torch.no_grad():
        output = model(val_x_tensor)
        if family == "frets":
            output = output[:, 0, 0]
        prediction = output.numpy() * target_std + target_mean
    if family == "frets":
        prediction = prediction * float(parameters["shrinkage_alpha"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "family": family, "state_dict": model.state_dict(), "parameters": parameters,
        "input_size": train_x.shape[-1], "target_mean": target_mean, "target_std": target_std,
        "source_sha256": None if source_path is None else sha256_file(source_path),
    }, checkpoint)
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return prediction.astype(float), {
        "training_seconds": time.perf_counter() - started, "epochs_completed": len(history),
        "best_validation_loss": best_loss, "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_predict_torch(
    checkpoint: Path, view: E5FoldView, source_path: Path | None,
) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    family, parameters = payload["family"], payload["parameters"]
    sequence_length = int(parameters["sequence_length"])
    feature_indices = [0] if family == "frets" else list(range(view.numeric_values.shape[-1]))
    x, _, _ = flattened_samples(view, "validation", sequence_length, feature_indices, False)
    if family == "frets":
        assert source_path is not None
        if sha256_file(source_path) != payload["source_sha256"]:
            raise RuntimeError("frozen FreTS source hash changed")
        module = load_frets_module(source_path, f"stage_e_e5_frets_load_{view.fold_id}")
        model = module.Model(SimpleNamespace(
            pred_len=1, enc_in=1, seq_len=sequence_length,
            channel_independence=str(parameters["channel_independence"]),
        ))
    else:
        model = MinimalistNetwork(int(payload["input_size"]), sequence_length, parameters)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(x.astype(np.float32)))
        if family == "frets":
            output = output[:, 0, 0]
        prediction = output.numpy() * float(payload["target_std"]) + float(payload["target_mean"])
    if family == "frets":
        prediction = prediction * float(parameters["shrinkage_alpha"])
    return prediction.astype(float)


def train_sklearn_model(
    family: str, view: E5FoldView, parameters: dict[str, Any], seed: int, checkpoint: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_x, train_y, _ = flattened_samples(view, "train", 12, list(range(view.numeric_values.shape[-1])), True)
    validation_x, _, _ = flattened_samples(view, "validation", 12, list(range(view.numeric_values.shape[-1])), False)
    train_x = train_x.reshape(len(train_x), -1)
    validation_x = validation_x.reshape(len(validation_x), -1)
    if family == "random_forest":
        model: Any = RandomForestRegressor(random_state=seed, **parameters)
    else:
        model = Pipeline([("scaler", StandardScaler()), ("svr", SVR(**parameters))])
    started = time.perf_counter()
    model.fit(train_x, train_y)
    prediction = model.predict(validation_x).astype(float)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, checkpoint)
    parameter_count = sum(tree.tree_.node_count for tree in model.estimators_) if family == "random_forest" else int(len(model.named_steps["svr"].support_))
    return prediction, {"training_seconds": time.perf_counter() - started, "parameter_count": parameter_count}


def load_predict_sklearn(checkpoint: Path, view: E5FoldView) -> np.ndarray:
    model = joblib.load(checkpoint)
    x, _, _ = flattened_samples(view, "validation", 12, list(range(view.numeric_values.shape[-1])), False)
    return model.predict(x.reshape(len(x), -1)).astype(float)


def train_industry_var(
    view: E5FoldView, industry_index: np.ndarray, industries: np.ndarray,
    alpha: float, checkpoint: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_indices = view.split_indices("train")
    validation_indices = view.split_indices("validation")
    industry_count = len(industries)

    def aggregate(indices: np.ndarray, targets: bool) -> np.ndarray:
        result = np.zeros((len(indices), industry_count), dtype=float)
        source = view.target_raw[indices] if targets else view.numeric_values[indices, -1, :, 0]
        valid = view.sample_mask[indices] if targets else view.node_available[indices]
        for industry in range(industry_count):
            stocks = industry_index == industry
            for row in range(len(indices)):
                mask = stocks & valid[row]
                result[row, industry] = float(source[row, mask].mean()) if mask.any() else 0.0
        return result

    x_train = aggregate(train_indices, False)
    y_train = aggregate(train_indices, True)
    design = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    x_validation = aggregate(validation_indices, False)
    industry_prediction = np.column_stack([np.ones(len(x_validation)), x_validation]) @ coefficients
    prediction = industry_prediction[:, industry_index].reshape(-1)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint, coefficients=coefficients, industry_index=industry_index, industries=industries)
    return prediction, {"training_seconds": 0.0, "parameter_count": int(coefficients.size)}


def load_predict_industry_var(checkpoint: Path, view: E5FoldView) -> np.ndarray:
    saved = np.load(checkpoint)
    coefficients = saved["coefficients"]
    industry_index = saved["industry_index"].astype(int)
    validation_indices = view.split_indices("validation")
    industry_count = coefficients.shape[1]
    current = np.zeros((len(validation_indices), industry_count), dtype=float)
    source = view.numeric_values[validation_indices, -1, :, 0]
    valid = view.node_available[validation_indices]
    for industry in range(industry_count):
        stocks = industry_index == industry
        for row in range(len(validation_indices)):
            mask = stocks & valid[row]
            current[row, industry] = float(source[row, mask].mean()) if mask.any() else 0.0
    industry_prediction = np.column_stack([np.ones(len(current)), current]) @ coefficients
    return industry_prediction[:, industry_index].reshape(-1)
