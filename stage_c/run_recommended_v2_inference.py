from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.bounded_ablations import minimalist_feature_view  # noqa: E402
from experiments.core import (  # noqa: E402
    DataBundle, environment_info, evaluate_predictions, prediction_frame, write_json,
)
from stage_c.inference import LoadedFixedEnsemble  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent inference for the recommended Stage C v2 ensemble.")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--data-root", type=Path,
        default=REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1",
    )
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify-reference", type=Path, default=None)
    args = parser.parse_args()

    manifest = args.manifest or (
        REPO_ROOT / "outputs/experiments/stage_c_30stocks_recommended_v2"
        / f"fixed_control_ensemble_v2_seed{args.seed}" / "model_manifest.json"
    )
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs/inference/stage_c_recommended_v2"
        / f"fixed_control_ensemble_v2_seed{args.seed}_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble = LoadedFixedEnsemble(manifest, REPO_ROOT, device=args.device)
    if ensemble.seed != args.seed:
        raise ValueError(f"manifest seed={ensemble.seed} does not match requested seed={args.seed}")
    data = DataBundle.load(args.data_root.resolve())
    data = minimalist_feature_view(data, "price_only")
    prediction, components = ensemble.predict(data, args.split)
    frame = prediction_frame(data.samples[args.split], prediction, args.split)
    for name, values in components.items():
        frame[f"component_{name}"] = values
    frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics = evaluate_predictions(frame)
    write_json(output_dir / "metrics.json", {
        "model": "fixed_control_ensemble_v2",
        "seed": args.seed,
        "split": args.split,
        "metrics": metrics,
        "sample_count": len(frame),
    })
    write_json(output_dir / "inference_provenance.json", {
        **ensemble.provenance(),
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "sample_count": len(frame),
        "environment": environment_info(REPO_ROOT),
    })

    verification = {"performed": False}
    if args.verify_reference is not None:
        reference = pd.read_csv(args.verify_reference)
        keys = ["stock_code", "target_date"]
        reconstructed = frame[keys + ["prediction"]].copy()
        reconstructed["target_date"] = pd.to_datetime(reconstructed["target_date"])
        reference["target_date"] = pd.to_datetime(reference["target_date"])
        aligned = reconstructed.merge(
            reference[keys + ["prediction"]], on=keys,
            suffixes=("_reconstructed", "_reference"), validate="one_to_one",
        )
        difference = np.abs(aligned["prediction_reconstructed"] - aligned["prediction_reference"])
        verification = {
            "performed": True,
            "reference": str(args.verify_reference.resolve()),
            "aligned_samples": len(aligned),
            "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "tolerance": 1e-7,
            "passed": bool(len(aligned) == len(frame) == len(reference) and difference.max() <= 1e-7),
        }
        if not verification["passed"]:
            write_json(output_dir / "verification.json", verification)
            raise AssertionError(f"independent inference did not reproduce reference: {verification}")
    write_json(output_dir / "verification.json", verification)
    print(json.dumps({
        "output_dir": str(output_dir),
        "metrics": metrics["aggregate"],
        "verification": verification,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
