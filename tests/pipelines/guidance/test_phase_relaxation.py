import unittest

import torch

from src.modeling.phases.relaxation import calc_phase_probs


class PredictSDSPhaseTest(unittest.TestCase):
    def test_calc_phase_probs_sums_to_one_over_phase_axis(self):
        probability = calc_phase_probs(
            torch.tensor([0.0, 1.0, 2.0]),
            num_phases=3,
            temperature=0.1,
            phase_dim=0,
        )

        self.assertTrue(torch.allclose(probability.sum(dim=0), torch.ones(3)))

    def test_calc_phase_probs_uses_squared_distance_logits(self):
        values = torch.tensor([0.25])
        temperature = 0.5

        probability = calc_phase_probs(
            values,
            num_phases=3,
            temperature=temperature,
            phase_dim=0,
        )

        levels = torch.tensor([0.0, 1.0, 2.0])
        expected = torch.softmax(-(values - levels).pow(2) / temperature, dim=0)
        self.assertTrue(torch.allclose(probability[:, 0], expected))

    def test_calc_phase_probs_rejects_non_floating_values(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            calc_phase_probs(torch.tensor([0, 1]), num_phases=4)

    def test_calc_phase_probs_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            calc_phase_probs(torch.tensor([float("nan")]), num_phases=2)


if __name__ == "__main__":
    unittest.main()
