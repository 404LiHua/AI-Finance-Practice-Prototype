from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stage_e.build_e4_adapter import align_text


class E4AdapterTests(unittest.TestCase):
    def test_missing_text_is_zero_without_dropping_sample(self) -> None:
        frame = pd.DataFrame([
            {"sample_row_id": "a", "text_available": True, "text_count": 1, "f1": 2.0},
            {"sample_row_id": "b", "text_available": False, "text_count": 0, "f1": 9.0},
        ])
        sample_ids = np.asarray([["a", "b"]], dtype="<U4")
        arrays = align_text(frame, ["f1"], {"a": (0, 0), "b": (0, 1)}, (1, 2), sample_ids)
        self.assertEqual(tuple(arrays["features"].shape), (1, 2, 1))
        self.assertEqual(float(arrays["features"][0, 1, 0]), 0.0)
        self.assertTrue(bool(arrays["sample_mask"].all()))


if __name__ == "__main__":
    unittest.main()
