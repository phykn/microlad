import unittest

import torch

from src.predict.sds.phase import soft_phase_probability


class PredictSDSPhaseTest(unittest.TestCase):
    def test_soft_phase_probability_sums_to_one_over_phase_axis(self):
        probability = soft_phase_probability(
            torch.tensor([-1.0, 0.0, 1.0]),
            num_phases=3,
            temperature=0.1,
            phase_dim=0,
        )

        self.assertTrue(torch.allclose(probability.sum(dim=0), torch.ones(3)))

    def test_soft_phase_probability_rejects_non_floating_values(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            soft_phase_probability(torch.tensor([0, 1]), num_phases=4)

    def test_soft_phase_probability_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            soft_phase_probability(torch.tensor([float("nan")]), num_phases=2)


if __name__ == "__main__":
    unittest.main()
