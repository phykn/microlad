import unittest

import torch

from src.predict.sds.phase import soft_phase_probability


class PredictSDSPhaseTest(unittest.TestCase):
    def test_soft_phase_probability_sums_to_one_over_phase_axis(self):
        probability = soft_phase_probability(
            torch.tensor([0.0, 1.0, 2.0]),
            num_phases=3,
            temperature=0.1,
            phase_dim=0,
        )

        self.assertTrue(torch.allclose(probability.sum(dim=0), torch.ones(3)))

    def test_soft_phase_probability_uses_squared_distance_logits(self):
        values = torch.tensor([0.25])
        temperature = 0.5

        probability = soft_phase_probability(
            values,
            num_phases=3,
            temperature=temperature,
            phase_dim=0,
        )

        levels = torch.tensor([0.0, 1.0, 2.0])
        expected = torch.softmax(-(values - levels).pow(2) / temperature, dim=0)
        self.assertTrue(torch.allclose(probability[:, 0], expected))

    def test_soft_phase_probability_rejects_non_floating_values(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            soft_phase_probability(torch.tensor([0, 1]), num_phases=4)

    def test_soft_phase_probability_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            soft_phase_probability(torch.tensor([float("nan")]), num_phases=2)


if __name__ == "__main__":
    unittest.main()
