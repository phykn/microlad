import unittest

import torch

from src.pipelines.guidance.metrics.conductance import ConductanceSolver
from src.pipelines.guidance.metrics.loss import descriptor_loss


class DescriptorLossTest(unittest.TestCase):
    def test_descriptor_loss_rejects_negative_weights(self):
        decoded = torch.zeros(2, 2)
        solver = ConductanceSolver(height=2, width=2)

        for kwargs in (
            {
                "fraction_targets": torch.tensor([0.5, 0.5]),
                "fraction_weight": -1.0,
            },
            {"tpc_targets": torch.zeros(2, 2), "tpc_weight": -1.0},
            {"sa_targets": torch.zeros(2), "sa_weight": -1.0},
            {
                "diffusivity_targets": torch.zeros(2),
                "diffusivity_solver": solver,
                "diffusivity_weight": -1.0,
            },
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "weight"):
                    descriptor_loss(decoded, num_phases=2, **kwargs)

    def test_descriptor_loss_requires_targets_when_weighted(self):
        decoded = torch.zeros(2, 2)

        for kwargs in (
            {"fraction_weight": 1.0},
            {"tpc_weight": 1.0},
            {"sa_weight": 1.0},
            {"diffusivity_weight": 1.0, "diffusivity_solver": ConductanceSolver(2, 2)},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "targets"):
                    descriptor_loss(decoded, num_phases=2, **kwargs)

    def test_descriptor_loss_allows_missing_targets_when_unweighted(self):
        decoded = torch.ones(2, 2)

        loss, stats = descriptor_loss(decoded, num_phases=2)

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))
        self.assertEqual(stats, {})

    def test_descriptor_loss_rejects_non_finite_decoded_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            descriptor_loss(torch.full((2, 2), float("nan")), num_phases=2)


if __name__ == "__main__":
    unittest.main()
