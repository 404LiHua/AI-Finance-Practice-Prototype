from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.core import DataBundle, write_json


class NaiveBaseline:
    name = "naive"

    def fit(self, data: DataBundle, output_dir: Path, logger: Any) -> None:
        write_json(output_dir / "model.json", {"model": self.name, "forecast_return": 0.0})

    def predict(self, data: DataBundle, split: str) -> np.ndarray:
        return np.zeros(len(data.samples[split]), dtype=float)


class MovingAverageBaseline:
    name = "moving_average"

    def __init__(self, window: int = 4) -> None:
        self.window = int(window)

    def fit(self, data: DataBundle, output_dir: Path, logger: Any) -> None:
        write_json(output_dir / "model.json", {"model": self.name, "window": self.window})

    def predict(self, data: DataBundle, split: str) -> np.ndarray:
        history = data.panel[["stock_code", "trade_date", "return_1w"]].copy()
        history["forecast"] = history.groupby("stock_code")["return_1w"].transform(
            lambda values: values.rolling(self.window, min_periods=1).mean()
        )
        sample = data.samples[split][["stock_code", "trade_date"]]
        merged = sample.merge(history, on=["stock_code", "trade_date"], how="left", validate="one_to_one")
        return merged["forecast"].fillna(0.0).to_numpy(dtype=float)


class ArimaBaseline:
    name = "arima"

    def __init__(
        self, order: list[int] | tuple[int, int, int], minimum_history: int = 12,
        maximum_history: int = 52, fallback_window: int = 4,
    ) -> None:
        self.order = tuple(int(v) for v in order)
        self.minimum_history = int(minimum_history)
        self.maximum_history = int(maximum_history)
        self.fallback_window = int(fallback_window)

    def fit(self, data: DataBundle, output_dir: Path, logger: Any) -> None:
        try:
            import statsmodels  # noqa: F401
            self.backend = "statsmodels"
        except ImportError:
            if self.order != (1, 0, 0):
                raise RuntimeError("The offline ARIMA backend currently supports order=(1,0,0) only")
            self.backend = "closed_form_ar1"
            logger.warning("statsmodels is unavailable; using the equivalent closed-form ARIMA(1,0,0) backend")
        write_json(output_dir / "model.json", {
            "model": self.name, "order": self.order,
            "minimum_history": self.minimum_history,
            "maximum_history": self.maximum_history,
            "fitting": "per-stock rolling one-step forecast",
            "backend": self.backend,
        })

    @staticmethod
    def _closed_form_ar1_forecast(series: np.ndarray) -> float:
        x = series[:-1]
        y = series[1:]
        design = np.column_stack([np.ones(len(x)), x])
        intercept, coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
        return float(intercept + coefficient * series[-1])

    def predict(self, data: DataBundle, split: str) -> np.ndarray:
        ARIMA = None
        if getattr(self, "backend", None) == "statsmodels":
            from statsmodels.tsa.arima.model import ARIMA

        forecasts: dict[tuple[str, pd.Timestamp], float] = {}
        wanted = set(zip(data.samples[split]["stock_code"], data.samples[split]["trade_date"]))
        for stock_code, stock_frame in data.panel.groupby("stock_code", sort=True):
            stock_frame = stock_frame.sort_values("trade_date")
            realized: list[float] = []
            for row in stock_frame.itertuples(index=False):
                value = getattr(row, "return_1w")
                if pd.notna(value):
                    realized.append(float(value))
                key = (stock_code, getattr(row, "trade_date"))
                if key not in wanted:
                    continue
                fallback = float(np.mean(realized[-self.fallback_window:])) if realized else 0.0
                if len(realized) < self.minimum_history:
                    forecasts[key] = fallback
                    continue
                series = np.asarray(realized[-self.maximum_history:], dtype=float)
                try:
                    if ARIMA is None:
                        forecast = self._closed_form_ar1_forecast(series)
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fitted = ARIMA(
                                series, order=self.order,
                                enforce_stationarity=False, enforce_invertibility=False,
                            ).fit()
                            forecast = float(fitted.forecast(1)[0])
                    forecasts[key] = forecast if np.isfinite(forecast) else fallback
                except Exception:
                    forecasts[key] = fallback
        return np.asarray([
            forecasts[(row.stock_code, row.trade_date)]
            for row in data.samples[split].itertuples(index=False)
        ], dtype=float)


class LSTMBaseline:
    name = "lstm"

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.model: Any = None
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.device = "cpu"

    def _prepare_scaler(self, data: DataBundle) -> None:
        train_rows = data.panel[data.panel["split"] == "train"][data.feature_columns]
        values = train_rows.to_numpy(dtype=np.float32)
        self.medians = np.nanmedian(values, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        filled = np.where(np.isfinite(values), values, self.medians)
        self.means = filled.mean(axis=0)
        self.stds = filled.std(axis=0)
        self.stds = np.where(self.stds < 1e-8, 1.0, self.stds)

    def _scaled_panel(self, data: DataBundle) -> pd.DataFrame:
        assert self.medians is not None and self.means is not None and self.stds is not None
        values = data.panel[data.feature_columns].to_numpy(dtype=np.float32)
        values = np.where(np.isfinite(values), values, self.medians)
        values = (values - self.means) / self.stds
        scaled = data.panel[["stock_code", "trade_date"]].copy()
        scaled["vector"] = list(values.astype(np.float32))
        return scaled

    def _sequences(self, data: DataBundle, split: str) -> tuple[np.ndarray, np.ndarray]:
        scaled = self._scaled_panel(data)
        seq_len = int(self.config["sequence_length"])
        lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
        for stock_code, frame in scaled.groupby("stock_code", sort=True):
            vectors = frame["vector"].tolist()
            dates = frame["trade_date"].tolist()
            for index, date in enumerate(dates):
                start = max(0, index - seq_len + 1)
                sequence = vectors[start:index + 1]
                if len(sequence) < seq_len:
                    sequence = [sequence[0]] * (seq_len - len(sequence)) + sequence
                lookup[(stock_code, date)] = np.stack(sequence)
        samples = data.samples[split]
        x = np.stack([lookup[(row.stock_code, row.trade_date)] for row in samples.itertuples(index=False)])
        y = samples["target_return"].to_numpy(dtype=np.float32)
        return x.astype(np.float32), y

    def fit(self, data: DataBundle, output_dir: Path, logger: Any) -> None:
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError("LSTM requires PyTorch. Install experiments/requirements-stage-b.txt") from exc

        self._prepare_scaler(data)
        train_x, train_y = self._sequences(data, "train")
        validation_x, validation_y = self._sequences(data, "validation")
        requested = str(self.config.get("device", "auto"))
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"

        class Network(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers=num_layers, batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, values: Any) -> Any:
                output, _ = self.lstm(values)
                return self.head(output[:, -1]).squeeze(-1)

        self.model = Network(
            train_x.shape[-1], int(self.config["hidden_size"]),
            int(self.config["num_layers"]), float(self.config["dropout"]),
        ).to(self.device)
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
        val_x_tensor = torch.from_numpy(validation_x).to(self.device)
        val_y_tensor = torch.from_numpy(validation_y).to(self.device)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, int(self.config["epochs"]) + 1):
            self.model.train()
            train_losses = []
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["gradient_clip"]))
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(val_x_tensor), val_y_tensor).cpu())
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss})
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % 10 == 0:
                logger.info("LSTM epoch=%d train_loss=%.8f validation_loss=%.8f", epoch, history[-1]["train_loss"], val_loss)
            if stale_epochs >= int(self.config["patience"]):
                logger.info("LSTM early stopping at epoch %d", epoch)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        torch.save({
            "state_dict": self.model.state_dict(), "config": self.config,
            "feature_columns": data.feature_columns, "medians": self.medians,
            "means": self.means, "stds": self.stds, "seed": self.seed,
        }, output_dir / "model.pt")
        write_json(output_dir / "training_history.json", history)

    def predict(self, data: DataBundle, split: str) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("LSTM has not been fitted")
        x, _ = self._sequences(data, split)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(x).to(self.device)).cpu().numpy().astype(float)


class MinimalistTransformerBaseline(LSTMBaseline):
    name = "minimalist_transformer"

    def fit(self, data: DataBundle, output_dir: Path, logger: Any) -> None:
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError("Minimalist Transformer requires PyTorch") from exc

        self._prepare_scaler(data)
        train_x, train_y = self._sequences(data, "train")
        validation_x, validation_y = self._sequences(data, "validation")
        requested = str(self.config.get("device", "auto"))
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"

        class Network(nn.Module):
            def __init__(self, input_size: int, sequence_length: int, cfg: dict[str, Any]) -> None:
                super().__init__()
                d_model = int(cfg["d_model"])
                self.input_projection = nn.Linear(input_size, d_model)
                self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=int(cfg["nhead"]),
                    dim_feedforward=int(cfg["dim_feedforward"]),
                    dropout=float(cfg["dropout"]),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=int(cfg["num_layers"]),
                    enable_nested_tensor=False,
                )
                self.output_norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, 1)
                nn.init.normal_(self.position, mean=0.0, std=0.02)

            def forward(self, values: Any) -> Any:
                encoded = self.input_projection(values) + self.position[:, :values.shape[1]]
                encoded = self.encoder(encoded)
                return self.head(self.output_norm(encoded[:, -1])).squeeze(-1)

        self.model = Network(train_x.shape[-1], train_x.shape[1], self.config).to(self.device)
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info("Minimalist Transformer parameters=%d device=%s", parameter_count, self.device)
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
        val_x_tensor = torch.from_numpy(validation_x).to(self.device)
        val_y_tensor = torch.from_numpy(validation_y).to(self.device)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, int(self.config["epochs"]) + 1):
            self.model.train()
            train_losses = []
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["gradient_clip"]))
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(val_x_tensor), val_y_tensor).cpu())
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss})
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % 10 == 0:
                logger.info(
                    "Minimalist Transformer epoch=%d train_loss=%.8f validation_loss=%.8f",
                    epoch, history[-1]["train_loss"], val_loss,
                )
            if stale_epochs >= int(self.config["patience"]):
                logger.info("Minimalist Transformer early stopping at epoch %d", epoch)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        torch.save({
            "state_dict": self.model.state_dict(), "config": self.config,
            "feature_columns": data.feature_columns, "medians": self.medians,
            "means": self.means, "stds": self.stds, "seed": self.seed,
            "parameter_count": parameter_count,
        }, output_dir / "model.pt")
        write_json(output_dir / "training_history.json", history)
        write_json(output_dir / "model_metadata.json", {
            "model": self.name,
            "parameter_count": parameter_count,
            "architecture": "linear projection + learned position + TransformerEncoder + final-token head",
        })


def build_model(name: str, config: dict[str, Any], seed: int) -> Any:
    if name == "naive":
        return NaiveBaseline()
    if name == "moving_average":
        return MovingAverageBaseline(**config["moving_average"])
    if name == "arima":
        return ArimaBaseline(**config["arima"])
    if name == "lstm":
        return LSTMBaseline(config["lstm"], seed)
    if name == "minimalist_transformer":
        return MinimalistTransformerBaseline(config["minimalist_transformer"], seed)
    raise ValueError(f"Unsupported model: {name}")
