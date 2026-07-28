from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from stage_f.custody import StageFDataCustodyGuard, StageFDataCustodyViolation
from stage_f.robustness import (
    FEATURE_MASKED_ID,
    F1_CANDIDATE_IDS,
    NOISE_AUGMENTED_ID,
    TAIL_WEIGHTED_ID,
    F1CandidateStrategy,
    build_f1_candidate_model,
    candidate_training_step,
    fit_train_tail_threshold,
)


ROOT = Path(__file__).resolve().parents[2]


def tiny_parameters() -> dict[str, object]:
    return {
        "sequence_length": 8,
        "hidden_channels": 8,
        "kernel_size": 3,
        "dilations": [1],
        "graph_order": 1,
        "dropout": 0.0,
        "batch_size": 4,
        "epochs": 80,
        "patience": 80,
        "learning_rate": 0.02,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "loss": "huber",
    }


class F1BoundedCandidateTest(unittest.TestCase):
    def test_deterministic_augmentation_and_inference_identity(self) -> None:
        values = torch.linspace(-1.0, 1.0, 4096).reshape(2, 8, 32, 8)
        for candidate_id in F1_CANDIDATE_IDS:
            left = F1CandidateStrategy(candidate_id, 20260725)
            right = F1CandidateStrategy(candidate_id, 20260725)
            left_first = left.transform(values, training=True)
            right_first = right.transform(values, training=True)
            left_second = left.transform(values, training=True)
            right_second = right.transform(values, training=True)
            torch.testing.assert_close(left_first, right_first, rtol=0.0, atol=0.0)
            torch.testing.assert_close(left_second, right_second, rtol=0.0, atol=0.0)
            torch.testing.assert_close(left.transform(values, training=False), values, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(
            F1CandidateStrategy(NOISE_AUGMENTED_ID, 20260725).transform(values, True), values,
        ))
        self.assertFalse(torch.equal(
            F1CandidateStrategy(FEATURE_MASKED_ID, 20260725).transform(values, True), values,
        ))

    def test_tensor_shapes_for_all_candidates(self) -> None:
        adjacency = np.eye(4, dtype=np.float32)
        values = torch.randn(3, 8, 4, 2)
        target = torch.randn(3, 4)
        mask = torch.ones_like(target, dtype=torch.bool)
        for candidate_id in F1_CANDIDATE_IDS:
            strategy = F1CandidateStrategy(candidate_id, 20260725)
            strategy.fit(target, mask, "train")
            transformed = strategy.transform(values, training=True)
            self.assertEqual(transformed.shape, values.shape)
            model = build_f1_candidate_model(candidate_id, 2, adjacency, tiny_parameters())
            prediction = model(transformed)
            self.assertEqual(prediction.shape, target.shape)
            self.assertEqual(strategy.loss(prediction, target, target, mask).ndim, 0)

    def test_minimum_overfit_for_all_candidates(self) -> None:
        adjacency = np.eye(4, dtype=np.float32)
        generator = torch.Generator().manual_seed(91)
        values = torch.randn((4, 8, 4, 2), generator=generator)
        target = 0.7 * values[:, -1, :, 0] - 0.2 * values[:, -1, :, 1]
        mask = torch.ones_like(target, dtype=torch.bool)
        for candidate_id in F1_CANDIDATE_IDS:
            np.random.seed(20260725)
            torch.manual_seed(20260725)
            model = build_f1_candidate_model(candidate_id, 2, adjacency, tiny_parameters())
            strategy = F1CandidateStrategy(candidate_id, 20260725)
            strategy.fit(target, mask, "train")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
            model.eval()
            with torch.no_grad():
                initial = float(torch.nn.functional.huber_loss(model(values)[mask], target[mask]))
            for _ in range(80):
                candidate_training_step(model, optimizer, strategy, values, target, target, mask, 1.0)
            model.eval()
            with torch.no_grad():
                final = float(torch.nn.functional.huber_loss(model(values)[mask], target[mask]))
            self.assertLess(final, initial * 0.35, msg=f"{candidate_id}: {initial} -> {final}")

    def test_train_only_fit_and_data_custody_boundaries(self) -> None:
        target = np.array([[0.01, -0.02], [0.20, -0.30]], dtype=np.float32)
        mask = np.ones_like(target, dtype=bool)
        threshold = fit_train_tail_threshold(target, mask, "train")
        self.assertGreater(threshold, 0.0)
        with self.assertRaises(ValueError):
            fit_train_tail_threshold(target, mask, "validation")
        for candidate_id in F1_CANDIDATE_IDS:
            strategy = F1CandidateStrategy(candidate_id, 20260725)
            with self.assertRaises(ValueError):
                strategy.fit(target, mask, "validation")
        guard = StageFDataCustodyGuard.from_config(ROOT / "stage_f/configs/f0_data_custody_v1.json", ROOT)
        for path in (
            ROOT / "outputs/stage_c/stage_c_recommended_v2_c4_20230609_20240607/predictions.csv",
            ROOT / "outputs/stage_d/d5_screening_20240614_20250613/predictions.csv",
            ROOT / "data/screening/panel.parquet",
            ROOT / "data/final/panel.parquet",
        ):
            with self.assertRaises(StageFDataCustodyViolation):
                guard.assert_path_allowed(path, "f1_1_minimal_test")
        with self.assertRaises(StageFDataCustodyViolation):
            guard.assert_development_dates(["2023-06-09"], "trade_date")


if __name__ == "__main__":
    unittest.main()
