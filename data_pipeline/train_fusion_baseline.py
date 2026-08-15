"""Train a reproducible tabular baseline on the fused weekly dataset."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "model_open", "model_high", "model_low", "model_close",
    "model_volume_hands", "model_amount_thousand_cny",
    "return_1w", "log_return_1w", "intraweek_range", "candle_body",
    "close_ma_4", "close_ma_12", "close_ma_26",
    "return_vol_4", "return_vol_12", "return_vol_26",
    "log_volume", "volume_z_12", "baostock_turnover_rate",
    "baostock_return_pct", "csmar_total_shares", "csmar_tradable_a_shares",
    "text_count", "weeks_since_listing",
]
CATEGORICAL_FEATURES = [
    "stock_code", "industry", "market_type", "enterprise_nature_current",
    "csmar_special_status", "text_available",
    "text_cluster",
]


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "direction_accuracy": float(np.mean((prediction > 0) == (y_true.to_numpy() > 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data_pipeline/configs/weekly_a_share.json"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args.data_root = args.data_root or Path(config["output"]["root"])
    if not args.data_root.is_absolute():
        args.data_root = project_root / args.data_root
    args.output = args.output or Path("outputs/stage_a_30stocks_baseline_v1")
    if not args.output.is_absolute():
        args.output = project_root / args.output
    train = pd.read_csv(args.data_root / "train.csv.gz")
    validation = pd.read_csv(args.data_root / "validation.csv.gz")
    test = pd.read_csv(args.data_root / "test.csv.gz")
    text_features = pd.read_csv(args.data_root / "text_features.csv.gz")
    text_features = text_features.drop(columns=["split", "text_count"], errors="ignore")
    for frame_name, frame in (("train", train), ("validation", validation), ("test", test)):
        merged = frame.merge(text_features, on=["stock_code", "calendar_week_end"], how="left", validate="one_to_one")
        if frame_name == "train":
            train = merged
        elif frame_name == "validation":
            validation = merged
        else:
            test = merged
    dynamic_text_features = sorted(column for column in train if column.startswith("text_svd_"))
    numeric_candidates = NUMERIC_FEATURES + dynamic_text_features
    feature_columns = [column for column in numeric_candidates + CATEGORICAL_FEATURES if column in train]
    numeric = [column for column in numeric_candidates if column in feature_columns]
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=3,
            random_state=20260723,
            n_jobs=-1,
        )),
    ])
    model.fit(train[feature_columns], train["target_return"])
    validation_prediction = model.predict(validation[feature_columns])
    test_prediction = model.predict(test[feature_columns])
    report = {
        "model": "RandomForestRegressor fusion baseline",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "feature_columns": feature_columns,
        "sample_counts": {
            "train": len(train), "validation": len(validation), "test": len(test),
        },
        "validation": metrics(validation["target_return"], validation_prediction),
        "test": metrics(test["target_return"], test_prediction),
        "limitations": [
            "Small thirty-stock one-year sample",
            "Sparse CSMAR special-treatment text; most text events are capital changes",
            "Baseline is not the proposed adaptive graph-frequency model",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    predictions = test[[
        "stock_code", "trade_date", "target_date", "target_return",
        "text_count", "csmar_special_status",
    ]].copy()
    predictions["prediction"] = test_prediction
    predictions["absolute_error"] = (predictions["prediction"] - predictions["target_return"]).abs()
    predictions.to_csv(args.output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
