from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FREEZE_CONFIG = REPO_ROOT / "stage_d/configs/d4_freeze.json"
FREEZE_DIR = REPO_ROOT / "stage_d/frozen/frets_l4_shrink_a075_d4"
CHECKPOINT_DIR = FREEZE_DIR / "checkpoints"
UPSTREAM_DIR = FREEZE_DIR / "upstream"
INFERENCE_MANIFEST = FREEZE_DIR / "INFERENCE_MANIFEST.json"
SHA_MANIFEST = FREEZE_DIR / "SHA256_MANIFEST.json"
SHA_SUMS = FREEZE_DIR / "SHA256SUMS"
RECEIPT = FREEZE_DIR / "FREEZE_RECEIPT.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_frozen_artifacts() -> dict[str, object]:
    freeze = json.loads(FREEZE_CONFIG.read_text(encoding="utf-8"))
    d2 = json.loads((REPO_ROOT / "stage_d/configs/d2_baselines.json").read_text(encoding="utf-8"))
    upstream_source = Path(d2["frets"]["root"]) / d2["frets"]["model_file"]
    if not upstream_source.is_file():
        raise FileNotFoundError(f"FreTS source missing before freeze: {upstream_source}")
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    frozen_source = UPSTREAM_DIR / "FreTS.py"
    shutil.copy2(upstream_source, frozen_source)
    source_sha = sha256_file(frozen_source)

    checkpoint_entries = []
    for seed in freeze["candidate"]["seeds"]:
        source_dir = REPO_ROOT / (
            f"outputs/stage_d/d2_bounded_baselines_v1/runs/"
            f"D_RO_03__frets_return_l4__seed{seed}"
        )
        target_dir = CHECKPOINT_DIR / f"seed{seed}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "resolved_config.json", "metrics.json", "seeds.json", "model_metadata.json"):
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"required D-4 source artifact missing: {source}")
            shutil.copy2(source, target_dir / name)
        checkpoint = target_dir / "model.pt"
        checkpoint_entries.append({
            "seed": int(seed),
            "path": relative(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        })
    manifest = {
        "freeze_id": freeze["freeze_id"],
        "candidate_model": freeze["candidate"]["model_id"],
        "base_model": freeze["candidate"]["base_model"],
        "checkpoint_fold": freeze["candidate"]["checkpoint_fold"],
        "feature_columns": freeze["candidate"]["feature_columns"],
        "sequence_length": freeze["candidate"]["sequence_length"],
        "shrinkage_alpha": freeze["candidate"]["shrinkage_alpha"],
        "aggregation": "arithmetic_mean",
        "aggregation_order": "apply shrinkage to each seed prediction, then average three seeds",
        "baseline": freeze["baseline"],
        "upstream_source": {
            "path": relative(frozen_source),
            "sha256": source_sha,
            "bytes": frozen_source.stat().st_size,
        },
        "checkpoints": checkpoint_entries,
    }
    write_json(INFERENCE_MANIFEST, manifest)
    return manifest


def collect_targets() -> dict[str, str]:
    targets = {}
    fixed = [
        (FREEZE_CONFIG, "freeze_policy"),
        (REPO_ROOT / "stage_d/configs/d3_diagnostics.json", "diagnostic_rule_source"),
        (REPO_ROOT / "stage_d/configs/d2_baselines.json", "candidate_registration"),
        (REPO_ROOT / "stage_d/protocols/rolling_origin_v1.json", "rolling_protocol"),
        (REPO_ROOT / "stage_d/configs/data_custody.json", "custody_policy"),
        (REPO_ROOT / "stage_d/inference.py", "independent_inference_code"),
        (REPO_ROOT / "stage_d/d4_policy.py", "frozen_decision_policy_code"),
        (REPO_ROOT / "stage_d/run_d4_independent_recalc.py", "independent_recalc_entry"),
        (REPO_ROOT / "stage_d/freeze_stage_d_candidate.py", "freeze_verification_code"),
        (REPO_ROOT / "experiments/core.py", "unified_evaluator_code"),
        (REPO_ROOT / "outputs/stage_d/d3_robust_diagnostics_v1/unique_candidate_recommendation.json", "selection_evidence"),
        (REPO_ROOT / "outputs/stage_d/d3_robust_diagnostics_v1/eligible_candidate_ranking.csv", "selection_evidence"),
        (REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1/metadata.json", "development_data_contract"),
        (REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1/selected_stocks.txt", "stock_universe"),
        (INFERENCE_MANIFEST, "inference_manifest"),
        (UPSTREAM_DIR / "FreTS.py", "frozen_upstream_source"),
    ]
    for path, role in fixed:
        if not path.is_file():
            raise FileNotFoundError(f"required freeze target missing: {path}")
        targets[relative(path)] = role
    for path in sorted(CHECKPOINT_DIR.rglob("*")):
        if path.is_file():
            targets[relative(path)] = "frozen_seed_checkpoint_artifact"
    return dict(sorted(targets.items()))


def build_entries(targets: dict[str, str]) -> list[dict[str, object]]:
    return [{
        "path": path_text,
        "role": role,
        "bytes": (REPO_ROOT / path_text).stat().st_size,
        "sha256": sha256_file(REPO_ROOT / path_text),
    } for path_text, role in targets.items()]


def build_freeze() -> dict[str, object]:
    freeze = json.loads(FREEZE_CONFIG.read_text(encoding="utf-8"))
    copy_frozen_artifacts()
    entries = build_entries(collect_targets())
    root_sha = canonical_digest(entries)
    write_json(SHA_MANIFEST, {
        "freeze_id": freeze["freeze_id"], "algorithm": "SHA-256",
        "repo_relative_paths": True, "artifact_count": len(entries), "entries": entries,
    })
    SHA_SUMS.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in entries), encoding="utf-8"
    )
    receipt = {
        "freeze_id": freeze["freeze_id"],
        "status": freeze["freeze_status"],
        "frozen_on": freeze["frozen_on"],
        "candidate": freeze["candidate"]["model_id"],
        "manifest_root_sha256": root_sha,
        "artifact_count": len(entries),
        "future_d_screening_status": freeze["future_d_screening_status"],
        "verification_command": "python -m stage_d.freeze_stage_d_candidate --verify",
        "independent_recalc_command": "python -m stage_d.run_d4_independent_recalc",
    }
    write_json(RECEIPT, receipt)
    return receipt


def verify_entries(entries: Iterable[dict[str, object]]) -> list[str]:
    errors = []
    for entry in entries:
        path = REPO_ROOT / str(entry["path"])
        if not path.is_file():
            errors.append(f"MISSING {entry['path']}")
            continue
        if path.stat().st_size != int(entry["bytes"]):
            errors.append(f"SIZE {entry['path']}")
        if sha256_file(path) != entry["sha256"]:
            errors.append(f"SHA256 {entry['path']}")
    return errors


def verify_freeze() -> dict[str, object]:
    manifest = json.loads(SHA_MANIFEST.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    errors = verify_entries(manifest["entries"])
    root_sha = canonical_digest(manifest["entries"])
    if root_sha != receipt["manifest_root_sha256"]:
        errors.append("MANIFEST_ROOT")
    if manifest["artifact_count"] != len(manifest["entries"]):
        errors.append("ARTIFACT_COUNT")
    result = {
        "freeze_id": manifest["freeze_id"], "verified": not errors,
        "artifact_count": len(manifest["entries"]), "manifest_root_sha256": root_sha,
        "errors": errors,
    }
    if errors:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the Stage D-4 candidate freeze.")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_freeze() if args.verify else build_freeze(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
