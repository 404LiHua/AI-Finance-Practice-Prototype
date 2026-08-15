"""Aggregate the complete Stage E-3 acceptance evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def batch_sha_valid(document: dict) -> bool:
    declared = document.get("batch_sha256", "")
    payload = dict(document)
    payload.pop("batch_sha256", None)
    return declared == stable_json_sha256(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e3_acceptance_v1.json")
    args = parser.parse_args()
    paths = {
        "graph_100": REPO_ROOT / "outputs/stage_e/e3_graph_acceptance_v1.json",
        "graph_300": REPO_ROOT / "outputs/stage_e/e3_graph_acceptance_300stocks_v1.json",
        "training": REPO_ROOT / "outputs/stage_e/e3_training_checks_v1/results.json",
        "scalability_300": REPO_ROOT / "outputs/stage_e/e3_scalability_300stocks_v1/results.json",
    }
    documents = {name: load(path) for name, path in paths.items()}
    training_root = paths["training"].parent
    scalability_root = paths["scalability_300"].parent
    training_artifacts = documents["training"]["artifacts"]
    checks = {
        "graph_100_pass": documents["graph_100"].get("passed") is True and documents["graph_100"].get("stock_count") == 100,
        "graph_300_pass": documents["graph_300"].get("passed") is True and documents["graph_300"].get("stock_count") == 300,
        "minimal_overfit_pass": documents["training"].get("overfit_pass") is True,
        "three_seed_stability_pass": documents["training"].get("stability_pass") is True,
        "training_batch_sha_valid": batch_sha_valid(documents["training"]),
        "scalability_300_pass": documents["scalability_300"].get("passed") is True,
        "scalability_batch_sha_valid": batch_sha_valid(documents["scalability_300"]),
        "training_curve_hash_valid": sha256_file(training_root / "training_curves.csv") == training_artifacts["training_curves_sha256"],
        "training_prediction_hash_valid": sha256_file(training_root / "seed_predictions.csv.gz") == training_artifacts["predictions_sha256"],
        "training_edge_hash_valid": sha256_file(training_root / "seed_edges.csv.gz") == training_artifacts["edges_sha256"],
        "training_checkpoint_hashes_valid": all(sha256_file(training_root / item["path"]) == item["sha256"] for item in training_artifacts["checkpoints"]),
        "scalability_edge_hash_valid": sha256_file(scalability_root / "adaptive_edges.csv.gz") == documents["scalability_300"]["artifacts"]["edges_sha256"],
        "scalability_checkpoint_hash_valid": sha256_file(scalability_root / "untrained_gradient_checked_model.pt") == documents["scalability_300"]["artifacts"]["checkpoint_sha256"],
        "future_or_sealed_data_not_read": not documents["training"].get("future_or_sealed_data_read", True) and not documents["scalability_300"].get("future_or_sealed_data_read", True),
    }
    report = {
        "stage": "E-3", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks,
        "evidence": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "summary": {
            "graph_100_shape": documents["graph_100"].get("rolling_graph_shape"),
            "graph_300_shape": documents["graph_300"].get("rolling_graph_shape"),
            "overfit_loss_reduction": documents["training"]["overfit"]["loss_reduction"],
            "minimum_edge_jaccard": min(item["edge_jaccard"] for item in documents["training"]["pairwise"]),
            "minimum_prediction_correlation": min(item["prediction_correlation"] for item in documents["training"]["pairwise"]),
            "scalability_adjacency_shape": documents["scalability_300"].get("adjacency_shape"),
        },
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
