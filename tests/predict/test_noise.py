import unittest

import torch

from src.predict.noise import guide_noise


class FractionAwareDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes = []
        self.axis_conditions = []

    def forward(self, x, t, phase_fractions=None, axis_condition=None):
        self.batch_sizes.append(x.shape[0])
        self.axis_conditions.append(
            None if axis_condition is None else axis_condition.detach().clone()
        )
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

    def test_guidance_repeats_axis_for_null_and_fraction_branches(self):
        model = FractionAwareDenoiser()
        patch = torch.zeros(2, 2, 4, 4)
        axes = torch.tensor([2, 2], dtype=torch.long)

        guide_noise(
            model,
            patch,
            torch.tensor([3, 3]),
            condition=torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
            axis_condition=axes,
            guidance=2.0,
        )

        self.assertEqual(len(model.axis_conditions), 1)
        self.assertTrue(
            torch.equal(
                model.axis_conditions[0],
                torch.tensor([2, 2, 2, 2], dtype=torch.long),
            )
        )

    def test_unconditional_fraction_path_still_forwards_axis(self):
        model = FractionAwareDenoiser()
        patch = torch.zeros(2, 2, 4, 4)
        axes = torch.tensor([1, 1], dtype=torch.long)

        guide_noise(
            model,
            patch,
            torch.tensor([3, 3]),
            condition=None,
            axis_condition=axes,
            guidance=1.0,
        )

        self.assertTrue(torch.equal(model.axis_conditions[0], axes))

if __name__ == "__main__":
    unittest.main()
