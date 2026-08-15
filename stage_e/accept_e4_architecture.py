"""Machine acceptance for the Stage E-4 graph-frequency architecture contract."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256
from stage_e.models.graph_frequency_fusion import GraphFrequencyFusionModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/stage_e/e4_architecture_acceptance_v1.json")
    args = parser.parse_args()
    torch.manual_seed(20260725)
    values = torch.randn(2, 12, 10, 6, requires_grad=True)
    available = torch.ones(2, 10, dtype=torch.bool)
    identity = torch.eye(10).unsqueeze(0).repeat(2, 1, 1)
    mode_results = []
    all_finite = True
    all_shapes = True
    gradients_positive = True
    for branch in ("temporal_only", "time_graph", "frequency_graph", "dual_branch"):
        for fusion in ("concat", "fixed_mean", "gated", "residual"):
            model = GraphFrequencyFusionModel(
                input_dim=6, stock_count=10, hidden_dim=16, top_k=4,
                graph_mode="learned_deterministic", branch_mode=branch, fusion_mode=fusion,
            )
            details = model(values, node_available=available, return_details=True)
            loss = details["prediction"].square().mean()
            gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True, allow_unused=True)
            gradient_norm = float(torch.sqrt(sum((gradient.detach() ** 2).sum() for gradient in gradients if gradient is not None)))
            finite = bool(torch.isfinite(details["prediction"]).all() and torch.isfinite(details["adjacency"]).all())
            shapes = tuple(details["prediction"].shape) == (2, 10) and tuple(details["adjacency"].shape) == (2, 10, 10)
            all_finite &= finite
            all_shapes &= shapes
            gradients_positive &= gradient_norm > 0
            mode_results.append({"branch": branch, "fusion": fusion, "finite": finite, "shapes_valid": shapes, "gradient_norm": gradient_norm})
    text_results = []
    text = torch.randn(2, 10, 7)
    for text_fusion in ("early", "mid"):
        model = GraphFrequencyFusionModel(
            input_dim=6, stock_count=10, hidden_dim=16, top_k=4,
            graph_mode="learned_deterministic", branch_mode="dual_branch", fusion_mode="fixed_mean",
            text_dim=7, text_fusion=text_fusion,
        )
        prediction = model(values, node_available=available, text_features=text)
        text_results.append({"text_fusion": text_fusion, "finite": bool(torch.isfinite(prediction).all()), "shape": list(prediction.shape)})
    provided_model = GraphFrequencyFusionModel(
        input_dim=6, stock_count=10, hidden_dim=16, top_k=4, graph_mode="provided", branch_mode="time_graph",
    )
    provided_details = provided_model(values, adjacency=identity, return_details=True)
    no_graph_model = GraphFrequencyFusionModel(
        input_dim=6, stock_count=10, hidden_dim=16, top_k=4, graph_mode="no_graph", branch_mode="dual_branch",
    )
    no_graph_details = no_graph_model(values, return_details=True)
    checks = {
        "all_branch_fusion_modes_finite": all_finite,
        "all_branch_fusion_shapes_valid": all_shapes,
        "all_branch_fusion_gradients_positive": gradients_positive,
        "provided_adjacency_preserved": bool(torch.equal(provided_details["adjacency"], identity)),
        "no_graph_uses_identity_only": bool(torch.equal(no_graph_details["adjacency"], identity)),
        "early_text_fusion_valid": text_results[0]["finite"] and text_results[0]["shape"] == [2, 10],
        "mid_text_fusion_valid": text_results[1]["finite"] and text_results[1]["shape"] == [2, 10],
    }
    model_path = REPO_ROOT / "stage_e/models/graph_frequency_fusion.py"
    report = {
        "stage": "E-4.1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()), "checks": checks, "mode_results": mode_results,
        "text_results": text_results, "input_shape": [2, 12, 10, 6],
        "temporal_axis_fft": 1, "stock_axis_graph": 2,
        "model_sha256": sha256_file(model_path), "future_or_sealed_data_read": False,
    }
    report["batch_sha256"] = stable_json_sha256(report)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
