from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "momentum_20d", "momentum_60d", "momentum_120d", "realized_volatility_20d", "realized_volatility_60d", "downside_volatility_60d", "current_drawdown_60d",
    "rsi_14", "macd_scaled", "bollinger_position_20", "amihud_20d", "zero_volume_fraction_20d", "volume_ratio_20d_60d", "intraday_range_mean_20d",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    features_path, predictions_path, policy_path, output = args.features.resolve(), args.predictions.resolve(), args.policy.resolve(), args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN_BEFORE_LABEL_FREE_SHADOW_AUDIT":
        raise RuntimeError("input-quality policy is not frozen")
    features = pd.read_parquet(features_path, engine="pyarrow"); predictions = pd.read_parquet(predictions_path, engine="pyarrow")
    for frame, label in ((features, "features"), (predictions, "predictions")):
        if frame.duplicated(["trade_date", "stock_code"]).any():
            raise RuntimeError(f"duplicate shadow identity: {label}")
        frame.trade_date = pd.to_datetime(frame.trade_date, errors="raise").dt.normalize()
    features.source_trade_date = pd.to_datetime(features.source_trade_date, errors="raise").dt.normalize()
    joined = features.merge(predictions, on=["trade_date", "stock_code"], how="left", validate="one_to_one", indicator=True)
    if (joined._merge != "both").any():
        raise RuntimeError("shadow prediction coverage failure")
    finite_features = np.isfinite(joined[FEATURES].to_numpy(float)).all(axis=1)
    probability = joined[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)
    finite_probability = np.isfinite(probability).all(axis=1) & np.isclose(probability.sum(axis=1), 1.0, atol=1e-12)
    joined["feature_age_days"] = (joined.trade_date - joined.source_trade_date).dt.days
    fresh = joined.feature_age_days <= int(policy["quality_gates"]["max_feature_age_days"])
    joined["operational_eligible"] = finite_features & finite_probability & fresh
    output.mkdir(parents=True)
    view_path = output / "LABEL_FREE_SHADOW_ELIGIBILITY_VIEW.parquet"
    joined[["trade_date", "stock_code", "source_trade_date", "feature_age_days", "operational_eligible", "t2_class", "prob_down", "prob_neutral", "prob_up", "confidence", "distribution_support", "reliability", "gated_ordinal_score"]].to_parquet(view_path, index=False, engine="pyarrow", compression="zstd")
    receipt = {
        "node_id": "AA_GFMNET_LABEL_FREE_SHADOW_INPUT_QUALITY_AUDIT_V1", "status": "PASS_SHADOW_INPUT_QUALITY_AUDITED_NO_LABEL_READ",
        "input_hashes": {"features": sha256(features_path), "predictions": sha256(predictions_path), "policy": sha256(policy_path)}, "eligibility_view_sha256": sha256(view_path),
        "rows": int(len(joined)), "feature_finite_rows": int(finite_features.sum()), "probability_contract_rows": int(finite_probability.sum()), "same_day_feature_rows": int(fresh.sum()), "operational_eligible_rows": int(joined.operational_eligible.sum()),
        "source_age_days": {"minimum": int(joined.feature_age_days.min()), "maximum": int(joined.feature_age_days.max()), "median": float(joined.feature_age_days.median())},
        "target_labels_read": False, "fresh_labels_read": False, "metrics_read": False, "model_trained": False, "gpu_used": False, "automatic_trading": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "SHADOW_INPUT_QUALITY_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "eligible_rows": receipt["operational_eligible_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


