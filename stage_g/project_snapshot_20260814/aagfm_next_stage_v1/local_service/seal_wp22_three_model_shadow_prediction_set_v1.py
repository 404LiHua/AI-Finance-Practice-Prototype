from __future__ import annotations

"""Seal aligned incumbent/C0/C1 label-free shadow predictions without labels."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["trade_date", "stock_code"]
PROBABILITY = ["prob_down", "prob_neutral", "prob_up"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(frame: pd.DataFrame) -> str:
    for column in ("model_id", "candidate_id"):
        if column in frame.columns and frame[column].nunique(dropna=False) == 1:
            return str(frame[column].iloc[0])
    raise RuntimeError("prediction file lacks a single model identity")


def validate(path: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_parquet(path)
    if not set(KEY + PROBABILITY).issubset(frame.columns):
        raise RuntimeError(f"prediction schema failure: {path}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    if frame.duplicated(KEY).any() or frame[KEY].isna().any().any():
        raise RuntimeError(f"duplicate/missing prediction key: {path}")
    values = frame[PROBABILITY].to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError(f"probability contract failure: {path}")
    return frame, model_identity(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True, type=Path)
    parser.add_argument("--c0", required=True, type=Path)
    parser.add_argument("--c1", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    paths = {"incumbent": args.incumbent.resolve(), "c0": args.c0.resolve(), "c1": args.c1.resolve()}
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output {output}")
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError("missing prediction input")
    validated = {label: validate(path) for label, path in paths.items()}
    expected = validated["incumbent"][0][KEY].sort_values(KEY, kind="mergesort").reset_index(drop=True)
    for label, (frame, _) in validated.items():
        actual = frame[KEY].sort_values(KEY, kind="mergesort").reset_index(drop=True)
        if not expected.equals(actual):
            raise RuntimeError(f"key coverage mismatch: {label}")
    if expected["trade_date"].nunique() != 1:
        raise RuntimeError("shadow predictions span more than one origin")
    output.mkdir(parents=True)
    manifest = {
        "node_id": "WP22_LABEL_FREE_THREE_MODEL_PREDICTION_SEAL_V1",
        "status": "SEALED_LABEL_FREE_SHADOW_PENDING_INDEPENDENT_T2_LABEL_AUTHORIZATION",
        "origin_date": expected["trade_date"].iloc[0].date().isoformat(), "rows": int(len(expected)),
        "models": {label: identity for label, (_, identity) in validated.items()},
        "input_sha256": {label: sha256(path) for label, path in paths.items()},
        "target_labels_read": False, "fresh_labels_read": False, "screening_read": False, "final_read": False,
        "metrics_read": False, "production_registry_modified": False, "automatic_trading": False,
        "production_replacement_allowed": False, "gpu_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output / "WP22_LABEL_FREE_THREE_MODEL_PREDICTION_SEAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {"node_id": manifest["node_id"], "status": manifest["status"], "manifest_sha256": sha256(manifest_path), "target_labels_read": False, "metrics_read": False, "gpu_used": False}
    (output / "WP22_THREE_MODEL_SHADOW_SEAL_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "rows": manifest["rows"], "models": manifest["models"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
