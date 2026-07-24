from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_CONFIG = REPO_ROOT / "stage_c/configs/recommended_v2_freeze_c3.json"
FREEZE_DIR = REPO_ROOT / "stage_c/frozen/recommended_v2_c3"
MANIFEST_PATH = FREEZE_DIR / "SHA256_MANIFEST.json"
SUMS_PATH = FREEZE_DIR / "SHA256SUMS"
RECEIPT_PATH = FREEZE_DIR / "FREEZE_RECEIPT.json"

SEEDS = (20260723, 20260724, 20260725)


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


def add_existing(targets: dict[str, str], path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required frozen artifact is missing: {path}")
    targets[relative(path)] = role


def add_run(targets: dict[str, str], run_dir: Path, role: str, require_model: bool = True) -> None:
    required = ["metrics.json", "resolved_config.json", "seeds.json", "predictions.csv"]
    if require_model:
        required.append("model.pt")
    else:
        required.append("model.json")
    for name in required:
        add_existing(targets, run_dir / name, role)
    for optional in ("model_metadata.json", "training_history.json", "environment.json"):
        path = run_dir / optional
        if path.is_file():
            add_existing(targets, path, role)


def collect_targets() -> dict[str, str]:
    targets: dict[str, str] = {}
    for path, role in (
        (FREEZE_CONFIG, "freeze_policy"),
        (REPO_ROOT / "stage_c/configs/recommended_v2.json", "candidate_config"),
        (REPO_ROOT / "stage_c/configs/stabilization_v1.json", "component_config"),
        (REPO_ROOT / "experiments/configs/stage_b_baselines.json", "baseline_config"),
        (REPO_ROOT / "experiments/configs/bounded_ablations.json", "baseline_config"),
        (REPO_ROOT / "stage_c/inference.py", "inference_code"),
        (REPO_ROOT / "stage_c/run_recommended_v2_inference.py", "inference_code"),
        (REPO_ROOT / "stage_c/build_recommended_v2.py", "candidate_build_code"),
        (REPO_ROOT / "stage_c/freeze_stage_c_candidate.py", "freeze_and_verification_code"),
        (REPO_ROOT / "stage_c/tests/test_candidate_freeze.py", "freeze_verification_test"),
        (REPO_ROOT / "experiments/core.py", "unified_evaluator_code"),
        (REPO_ROOT / "experiments/models.py", "baseline_model_code"),
        (REPO_ROOT / "experiments/bounded_ablations.py", "baseline_model_code"),
        (REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1/metadata.json", "development_data_contract"),
        (REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1/selected_stocks.txt", "stock_universe"),
        (REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1/train.csv.gz", "development_train_split"),
        (REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1/validation.csv.gz", "development_validation_split"),
        (REPO_ROOT / "outputs/experiments/stage_c_30stocks_recommended_v2/recommended_v2_summary.csv", "candidate_development_result"),
        (REPO_ROOT / "outputs/experiments/stage_c_30stocks_recommended_v2/recommended_v2_results.csv", "candidate_development_result"),
        (REPO_ROOT / "outputs/experiments/stage_c_30stocks_recommended_v2/recommended_v2_unified_comparison.csv", "frozen_comparison"),
        (REPO_ROOT / "outputs/experiments/stage_c_30stocks_recommended_v2/recommended_v2_decision.json", "candidate_decision"),
    ):
        add_existing(targets, path, role)

    for path in sorted((REPO_ROOT / "stage_c/models").glob("*.py")):
        add_existing(targets, path, "candidate_model_code")

    recommendation_root = REPO_ROOT / "outputs/experiments/stage_c_30stocks_recommended_v2"
    stabilization_root = REPO_ROOT / "outputs/experiments/stage_c_30stocks_graph_stabilization"
    for seed in SEEDS:
        candidate_dir = recommendation_root / f"fixed_control_ensemble_v2_seed{seed}"
        for name in ("model_manifest.json", "metrics.json", "predictions.csv"):
            add_existing(targets, candidate_dir / name, "candidate_seed_artifact")
        add_run(targets, stabilization_root / f"temporal_only_control_seed{seed}", "candidate_component")
        add_run(targets, stabilization_root / f"fixed_temporal_graph_control_seed{seed}", "candidate_component")

        add_run(
            targets,
            REPO_ROOT / f"outputs/experiments/stage_b_30stocks_baselines/naive_seed{seed}",
            "frozen_baseline_naive",
            require_model=False,
        )
        add_run(
            targets,
            REPO_ROOT / f"outputs/experiments/stage_b_30stocks_bounded_ablations/frets_return_l4_seed{seed}",
            "frozen_baseline_frets",
        )
        add_run(
            targets,
            REPO_ROOT / f"outputs/experiments/stage_b_30stocks_bounded_ablations/minimalist_price_only_l8_seed{seed}",
            "frozen_baseline_minimalist_transformer",
        )
        add_run(
            targets,
            REPO_ROOT / f"outputs/experiments/stage_c_graph_frequency_v1/graph_frequency_v1_seed{seed}",
            "frozen_baseline_graph_frequency_v1",
        )
    return dict(sorted(targets.items()))


def build_entries(targets: dict[str, str]) -> list[dict[str, object]]:
    entries = []
    for path_text, role in targets.items():
        path = REPO_ROOT / path_text
        entries.append({
            "path": path_text,
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return entries


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_freeze() -> dict[str, object]:
    config = json.loads(FREEZE_CONFIG.read_text(encoding="utf-8"))
    entries = build_entries(collect_targets())
    manifest = {
        "freeze_id": config["freeze_id"],
        "algorithm": "SHA-256",
        "repo_relative_paths": True,
        "artifact_count": len(entries),
        "entries": entries,
    }
    root_digest = canonical_digest(entries)
    receipt = {
        "freeze_id": config["freeze_id"],
        "status": config["freeze_status"],
        "frozen_on": config["frozen_on"],
        "manifest_root_sha256": root_digest,
        "artifact_count": len(entries),
        "screening_data_status": config["screening_data_status"],
        "verification_command": "python -m stage_c.freeze_stage_c_candidate --verify",
    }
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST_PATH, manifest)
    SUMS_PATH.write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries),
        encoding="utf-8",
    )
    write_json(RECEIPT_PATH, receipt)
    return receipt


def verify_entries(entries: Iterable[dict[str, object]]) -> list[str]:
    errors = []
    for entry in entries:
        path = REPO_ROOT / str(entry["path"])
        if not path.is_file():
            errors.append(f"MISSING {entry['path']}")
            continue
        actual_size = path.stat().st_size
        if actual_size != int(entry["bytes"]):
            errors.append(f"SIZE {entry['path']}: expected={entry['bytes']} actual={actual_size}")
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            errors.append(f"SHA256 {entry['path']}: expected={entry['sha256']} actual={actual_hash}")
    return errors


def verify_freeze() -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    errors = verify_entries(manifest["entries"])
    actual_root = canonical_digest(manifest["entries"])
    if actual_root != receipt["manifest_root_sha256"]:
        errors.append(
            "MANIFEST_ROOT: "
            f"expected={receipt['manifest_root_sha256']} actual={actual_root}"
        )
    if int(manifest["artifact_count"]) != len(manifest["entries"]):
        errors.append("ARTIFACT_COUNT mismatch")
    result = {
        "freeze_id": manifest["freeze_id"],
        "verified": not errors,
        "artifact_count": len(manifest["entries"]),
        "manifest_root_sha256": actual_root,
        "errors": errors,
    }
    if errors:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the Stage C recommended-v2 freeze.")
    parser.add_argument("--verify", action="store_true", help="verify frozen hashes without modifying files")
    args = parser.parse_args()
    result = verify_freeze() if args.verify else build_freeze()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
