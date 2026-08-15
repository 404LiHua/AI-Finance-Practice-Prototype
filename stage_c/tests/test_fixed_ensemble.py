import unittest

import torch

from stage_c.models import FixedEqualEnsemble


class FixedEqualEnsembleTest(unittest.TestCase):
    def test_equal_average_and_shape_guard(self) -> None:
        model = FixedEqualEnsemble()
        temporal = torch.tensor([0.1, -0.2])
        fixed = torch.tensor([0.3, 0.0])
        self.assertTrue(torch.allclose(model(temporal, fixed), torch.tensor([0.2, -0.1])))
        with self.assertRaises(ValueError):
            model(temporal, fixed[:1])


if __name__ == "__main__":
    unittest.main()

