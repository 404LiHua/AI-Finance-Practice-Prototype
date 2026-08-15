from __future__ import annotations

"""Audit the sealed WP11 C0/incumbent shadow pair without opening labels."""

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
KEY = ["trade_date", "stock_code"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_prediction(path: Path, model_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    if frame.duplicated(KEY).any() or frame.model_id.nunique() != 1 or frame.model_id.iloc[0] != model_id:
        raise RuntimeError(f"shadow identity/model contract failure: {model_id}")
    probabilities = frame[["prob_down", "prob_neutral", "prob_up"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError(f"shadow probability contract failure: {model_id}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--incumbent-predictions", required=True, type=Path)
    parser.add_argument("--c0-predictions", required=True, type=Path)
    parser.add_argument("--seal-manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    feature_path, incumbent_path, c0_path, seal_path, protocol_path, output = (args.features.resolve(), args.incumbent_predictions.resolve(), args.c0_predictions.resolve(), args.seal_manifest.resolve(), args.protocol.resolve(), args.output_root.resolve())
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_C0_ARTIFACT_MATERIALIZATION_OR_WP11_SHADOW_PREDICTION" or seal.get("status") != "SEALED_LABEL_FREE_SHADOW_PENDING_FUTURE_INDEPENDENT_AUTHORIZATION":
        raise RuntimeError("WP11 protocol/seal state failure")
    identities = protocol["input_identities"]
    required_hashes = {"features": feature_path, "incumbent_predictions": incumbent_path, "c0_predictions": c0_path}
    for role, path in required_hashes.items():
        expected = seal["input_sha256"].get(role)
        if sha256(path) != expected:
            raise RuntimeError(f"sealed input hash mismatch: {role}")
    if sha256(feature_path) != identities["label_free_feature_sha256"] or sha256(incumbent_path) != identities["incumbent_shadow_prediction_sha256"]:
        raise RuntimeError("frozen WP11 source identity mismatch")
    features = pd.read_parquet(feature_path, engine="pyarrow")
    features.trade_date = pd.to_datetime(features.trade_date, errors="raise").dt.normalize(); features.source_trade_date = pd.to_datetime(features.source_trade_date, errors="raise").dt.normalize()
    incumbent = checked_prediction(incumbent_path, protocol["governance"]["active_production_model"])
    c0 = checked_prediction(c0_path, protocol["governance"]["candidate_id"])
    for frame in (incumbent, c0):
        frame.trade_date = pd.to_datetime(frame.trade_date, errors="raise").dt.normalize()
    if features.duplicated(KEY).any() or len(features) != int(protocol["shadow"]["frozen_universe_size"]):
        raise RuntimeError("shadow feature key/cardinality failure")
    base = features[KEY + ["source_trade_date", *FEATURES]].merge(incumbent[KEY + ["prob_down", "prob_neutral", "prob_up", "reliability", "gated_ordinal_score"]], on=KEY, how="left", validate="one_to_one", indicator="incumbent_join")
    base = base.merge(c0[KEY + ["prob_down", "prob_neutral", "prob_up", "candidate_ordinal_score"]], on=KEY, how="left", validate="one_to_one", suffixes=("_incumbent", "_c0"), indicator="c0_join")
    if (base.incumbent_join != "both").any() or (base.c0_join != "both").any():
        raise RuntimeError("two-model shadow coverage failure")
    feature_finite = np.isfinite(base[FEATURES].to_numpy(float)).all(axis=1)
    base["feature_age_days"] = (base.trade_date - base.source_trade_date).dt.days
    c0_probability = base[["prob_down_c0", "prob_neutral_c0", "prob_up_c0"]].to_numpy(float)
    incumbent_probability = base[["prob_down_incumbent", "prob_neutral_incumbent", "prob_up_incumbent"]].to_numpy(float)
    base["operational_eligible"] = feature_finite & base.feature_age_days.eq(0) & np.isfinite(c0_probability).all(axis=1) & np.isfinite(incumbent_probability).all(axis=1)
    max_abs_delta = float(np.max(np.abs(c0_probability - incumbent_probability)))
    same_probability_rows = int(np.isclose(c0_probability, incumbent_probability, atol=1e-12, rtol=0.0).all(axis=1).sum())
    output.mkdir(parents=True)
    view_columns = KEY + ["source_trade_date", "feature_age_days", "operational_eligible", "reliability", "gated_ordinal_score", "candidate_ordinal_score", "prob_down_incumbent", "prob_neutral_incumbent", "prob_up_incumbent", "prob_down_c0", "prob_neutral_c0", "prob_up_c0"]
    view_path = output / "WP11_TWO_MODEL_LABEL_FREE_ELIGIBILITY_VIEW.parquet"
    base[view_columns].to_parquet(view_path, index=False, engine="pyarrow", compression="zstd")
    receipt = {"node_id": "AA_GFMNET_WP11_TWO_MODEL_SHADOW_QUALITY_AUDIT_V1", "status": "PASS_TWO_MODEL_LABEL_FREE_SHADOW_QUALITY_AUDITED", "protocol_sha256": sha256(protocol_path), "seal_manifest_sha256": sha256(seal_path), "input_hashes": {role: sha256(path) for role, path in required_hashes.items()}, "eligibility_view_sha256": sha256(view_path), "rows": int(len(base)), "finite_feature_rows": int(feature_finite.sum()), "same_day_feature_rows": int(base.feature_age_days.eq(0).sum()), "operational_eligible_rows": int(base.operational_eligible.sum()), "max_abs_probability_difference_between_models": max_abs_delta, "identical_probability_rows_between_models": same_probability_rows, "target_labels_read": False, "fresh_labels_read": False, "wp10_outputs_read": False, "metrics_read": False, "model_trained": False, "gpu_used": False, "automatic_trading": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "WP11_TWO_MODEL_SHADOW_QUALITY_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "operational_eligible_rows": receipt["operational_eligible_rows"], "max_abs_probability_difference_between_models": max_abs_delta}, ensure_ascii=False))


if __name__ == "__main__":
    main()


