import unittest

import torch

from stage_c.models.frequency_graph import FrequencyGraphBlock


class FrequencyGraphTest(unittest.TestCase):
    def test_fft_ifft_reconstruction_and_block_shape(self) -> None:
        torch.manual_seed(11)
        values = torch.randn(3, 8, 12)
        reconstructed = torch.fft.ifft(
            torch.fft.fft(values, dim=1, norm="ortho"), dim=1, norm="ortho",
        ).real
        self.assertTrue(torch.allclose(values, reconstructed, atol=1e-5))
        adjacency = torch.eye(8).unsqueeze(0).repeat(3, 1, 1)
        output = FrequencyGraphBlock(hidden_dim=12)(values, adjacency)
        self.assertEqual(tuple(output.shape), tuple(values.shape))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

