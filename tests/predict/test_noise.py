import unittest

import torch

from src.predict.noise import guide_noise


class FractionAwareDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes = []

    def forward(self, x, t, phase_fractions=None):
        self.batch_sizes.append(x.shape[0])
        if phase_fractions is None:
            return torch.zeros_like(x)
        value = phase_fractions[:, :1, None, None]
        return torch.ones_like(x) * value


class NoiseTest(unittest.TestCase):
    def test_guidance_combines_null_and_fraction_predictions(self):
        model = FractionAwareDenoiser()
        patch = torch.zeros(2, 2, 4, 4)
        condition = torch.tensor([[0.25, 0.75], [0.5, 0.5]])

        noise = guide_noise(
            model,
            patch,
            torch.tensor([3, 3]),
            condition=condition,
            guidance=2.0,
        )

        expected = torch.tensor([0.5, 1.0]).view(2, 1, 1, 1).expand_as(noise)
        self.assertTrue(torch.equal(noise, expected))
        self.assertEqual(model.batch_sizes, [4])


if __name__ == "__main__":
    unittest.main()
