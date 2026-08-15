from __future__ import annotations

"""Build current, label-free RG3 inputs for the active local T2 model."""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SELECTED_SHA256 = "7522a6053cd143f0046895713b0f66f76a30b15d9ff8ebb8410dc27b0da67f5c"
RG3_SOURCE_SHA256 = "68d9091006565c6a454c731311bd7ddc65af4112624f40681f7df1a82f23d584"
INCUMBENT_LAST_FIT_ORIGIN = pd.Timestamp("2026-06-26")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rg3(path: Path, root: Path):
    sys.dont_write_bytecode = True
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("shadow_rg3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load frozen RG3 feature source")
    module = importlib.util.module_from_spec(spec); sys.modules["shadow_rg3"] = module; spec.loader.exec_module(module)
    return module


def read_daily_prefix(path: Path, origin: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[1, 2, 4, 5, 9, 10, 11])
    frame.columns = ["trade_date", "close", "high", "low", "volume", "amount", "adjust_factor"]
    frame.trade_date = pd.to_datetime(frame.trade_date, errors="coerce").dt.normalize()
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[frame.trade_date.notna() & (frame.trade_date <= origin)].sort_values("trade_date", kind="mergesort").drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--daily-root", type=Path, required=True)
    parser.add_argument("--selected-universe", type=Path, required=True)
    parser.add_argument("--origin-date", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source, daily_root, selected_path, output = args.source_root.resolve(), args.daily_root.resolve(), args.selected_universe.resolve(), args.output_root.resolve()
    origin = pd.Timestamp(args.origin_date).normalize()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    if origin <= INCUMBENT_LAST_FIT_ORIGIN or origin.weekday() != 4:
        raise RuntimeError("shadow origin must be a Friday strictly later than the incumbent fit cutoff")
    if not selected_path.is_file() or sha256(selected_path) != SELECTED_SHA256:
        raise RuntimeError("selected-universe identity mismatch")
    rg3_path = source / "src" / "rg3_features.py"
    if not rg3_path.is_file() or sha256(rg3_path) != RG3_SOURCE_SHA256:
        raise RuntimeError("frozen RG3 source identity mismatch")
    rg3 = load_rg3(rg3_path, source)
    selected = pd.read_csv(selected_path, usecols=["stock_code", "selection_rank"], dtype={"stock_code": str}).sort_values("selection_rank", kind="mergesort")
    if len(selected) != 300 or selected.stock_code.duplicated().any():
        raise RuntimeError("expected the frozen 300-stock selected universe")
    rows, daily_hashes = [], {}
    for code in selected.stock_code:
        path = daily_root / f"{code}.csv"
        if not path.is_file():
            raise RuntimeError(f"missing daily source: {code}")
        daily = read_daily_prefix(path, origin)
        features = rg3.build_daily_technical_features(daily)
        aligned = rg3.align_daily_features_to_decision_dates(features, [origin])
        row = aligned.iloc[0].to_dict(); row["stock_code"] = code
        rows.append(row); daily_hashes[code] = sha256(path)
    result = pd.DataFrame(rows)
    columns = list(rg3.DAILY_TECHNICAL_FEATURES)
    if result.source_trade_date.isna().any() or (result.source_trade_date > origin).any() or result[columns].isna().any().any() or not np.isfinite(result[columns].to_numpy(float)).all():
        raise RuntimeError("label-free RG3 shadow feature contract failure")
    result["trade_date"] = origin
    result = result[["trade_date", "stock_code", "source_trade_date", *columns]].sort_values("stock_code", kind="mergesort")
    output.mkdir(parents=True)
    feature_path = output / "T2_LABEL_FREE_SHADOW_FEATURES.parquet"; result.to_parquet(feature_path, index=False, engine="pyarrow", compression="zstd")
    receipt = {
        "node_id": "AA_GFMNET_T2_LABEL_FREE_SHADOW_FEATURES_V1", "status": "PASS_LABEL_FREE_SHADOW_INPUT_SEALED",
        "origin_date": origin.date().isoformat(), "rows": int(len(result)), "feature_count": len(columns),
        "selected_universe_sha256": sha256(selected_path), "rg3_source_sha256": sha256(rg3_path), "daily_source_sha256": daily_hashes,
        "feature_sha256": sha256(feature_path), "source_trade_date_min": pd.Timestamp(result.source_trade_date.min()).date().isoformat(), "source_trade_date_max": pd.Timestamp(result.source_trade_date.max()).date().isoformat(),
        "target_labels_read": False, "fresh_labels_read": False, "metrics_read": False, "model_trained": False, "gpu_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "SHADOW_FEATURE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output_root": str(output), "rows": receipt["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


