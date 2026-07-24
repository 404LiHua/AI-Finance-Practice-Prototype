from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import prediction_frame, write_json  # noqa: E402
from stage_d.d2_baselines import build_fold_bundle, load_locked_config, validate_protocol  # noqa: E402
from stage_d.d4_policy import evaluate_frozen_policy  # noqa: E402
from stage_d.freeze_stage_d_candidate import FREEZE_CONFIG, FREEZE_DIR, verify_freeze  # noqa: E402
from stage_d.inference import LoadedStageDFrozenCandidate, sha256_file  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "outputs/stage_d/d4_independent_recalc_v1"


def run() -> dict:
    verification = verify_freeze()
    freeze = json.loads(FREEZE_CONFIG.read_text(encoding="utf-8"))
    d3 = json.loads((REPO_ROOT / "stage_d/configs/d3_diagnostics.json").read_text(encoding="utf-8"))
    d2 = load_locked_config(REPO_ROOT / "stage_d/configs/d2_baselines.json", REPO_ROOT)
    protocol = validate_protocol(d2)
    data, fold_evidence = build_fold_bundle(d2, protocol, "D_RO_03", REPO_ROOT)
    loader = LoadedStageDFrozenCandidate(FREEZE_DIR / "INFERENCE_MANIFEST.json", REPO_ROOT)
    aggregate, per_seed = loader.predict(data, "validation")

    keys = ["stock_code", "trade_date", "target_date"]
    stored_seed_frames = []
    maximum_abs_difference = 0.0
    for seed, prediction in per_seed.items():
        path = REPO_ROOT / (
            "outputs/stage_d/d2_bounded_baselines_v1/derived/"
            f"D_RO_03__frets_return_l4__fixed_shrink_a075__seed{seed}/predictions.csv"
        )
        stored = pd.read_csv(path)
        expected = stored["prediction"].to_numpy(float)
        difference = float(np.max(np.abs(prediction - expected)))
        maximum_abs_difference = max(maximum_abs_difference, difference)
        if difference > 1e-7:
            raise RuntimeError(f"independent checkpoint recomputation differs for seed {seed}: {difference}")
        seed_frame = stored[keys].copy()
        seed_frame[f"seed_{seed}_prediction"] = expected
        stored_seed_frames.append(seed_frame)
    aligned = stored_seed_frames[0]
    for frame in stored_seed_frames[1:]:
        aligned = aligned.merge(frame, on=keys, how="inner", validate="one_to_one")
    stored_aggregate = aligned.filter(like="seed_").mean(axis=1).to_numpy(float)
    aggregate_difference = float(np.max(np.abs(aggregate - stored_aggregate)))
    if aggregate_difference > 1e-7:
        raise RuntimeError("independent aggregate recomputation differs from frozen rule")

    decision = evaluate_frozen_policy(
        data.samples["validation"], aggregate, per_seed, freeze, d3["return_groups"]
    )
    decision["scope"] = "D_RO_03_DEVELOPMENT_RECALC_NOT_SCREENING"
    decision["screening_decision_valid"] = False
    decision["development_recalc_outcome_for_diagnostics_only"] = decision.pop("outcome")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = prediction_frame(data.samples["validation"], aggregate, "validation")
    for seed, values in per_seed.items():
        frame[f"seed_{seed}_prediction"] = values
    frame["naive_prediction"] = 0.0
    frame.to_csv(OUTPUT_ROOT / "independent_recalc_predictions.csv", index=False, encoding="utf-8-sig")
    write_json(OUTPUT_ROOT / "independent_recalc_result.json", {
        "freeze_verification": verification,
        "fold_evidence": fold_evidence,
        "per_seed_maximum_absolute_difference": maximum_abs_difference,
        "aggregate_maximum_absolute_difference": aggregate_difference,
        "decision_diagnostics": decision,
        "c4_rows_read": 0,
        "future_d_screening_rows_read": 0,
    })
    receipt = {
        "freeze_id": freeze["freeze_id"],
        "independent_load": "PASS",
        "independent_prediction_recalc": "PASS",
        "per_seed_maximum_absolute_difference": maximum_abs_difference,
        "aggregate_maximum_absolute_difference": aggregate_difference,
        "prediction_sha256": sha256_file(OUTPUT_ROOT / "independent_recalc_predictions.csv"),
        "result_sha256": sha256_file(OUTPUT_ROOT / "independent_recalc_result.json"),
        "c4_rows_read": 0,
        "future_d_screening_rows_read": 0,
    }
    write_json(OUTPUT_ROOT / "INDEPENDENT_RECALC_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
