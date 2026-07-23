from __future__ import annotations

from typing import Any

from experiments.core import DataBundle
from experiments.external_adapters import FreTSAdapter
from experiments.models import MinimalistTransformerBaseline


PRICE_FEATURES = [
    "model_open",
    "model_high",
    "model_low",
    "model_close",
    "return_1w",
    "log_return_1w",
    "intraweek_range",
    "candle_body",
    "close_ma_4",
    "close_ma_12",
    "close_ma_26",
    "return_vol_4",
    "return_vol_12",
    "return_vol_26",
]


def require_features(data: DataBundle, feature_columns: list[str]) -> list[str]:
    missing = [column for column in feature_columns if column not in data.panel.columns]
    if missing:
        raise KeyError(f"Ablation requires missing features: {missing}")
    return feature_columns


def minimalist_feature_view(data: DataBundle, feature_set: str) -> DataBundle:
    price = require_features(data, PRICE_FEATURES.copy())
    if feature_set == "price_only":
        selected = price
    elif feature_set == "price_text":
        text = [
            column for column in data.feature_columns
            if column == "text_count" or column.startswith("text_svd_")
        ]
        if not text:
            raise KeyError("Price+text ablation requires train-fitted text features")
        selected = price + text
    else:
        raise ValueError(f"Unsupported Minimalist Transformer feature set: {feature_set}")
    return DataBundle(panel=data.panel, samples=data.samples, feature_columns=selected)


class FreTSBoundedAblationAdapter(FreTSAdapter):
    def __init__(
        self, config: dict[str, Any], external: dict[str, Any], seed: int,
        variant_id: str,
    ) -> None:
        super().__init__(config, external, seed)
        self.name = variant_id

    def select_features(self, data: DataBundle) -> list[str]:
        feature_set = str(self.config["feature_set"])
        if feature_set == "return_only":
            return require_features(data, ["return_1w"])
        if feature_set == "return_ohlc":
            # The upstream FreTS returns one forecast per channel. Keeping the
            # return channel first makes the unified adapter select its output.
            return require_features(
                data,
                ["return_1w", "model_open", "model_high", "model_low", "model_close"],
            )
        raise ValueError(f"Unsupported FreTS feature set: {feature_set}")


class MinimalistTransformerFeatureAblation(MinimalistTransformerBaseline):
    def __init__(self, config: dict[str, Any], seed: int, variant_id: str) -> None:
        super().__init__(config, seed)
        self.name = variant_id
