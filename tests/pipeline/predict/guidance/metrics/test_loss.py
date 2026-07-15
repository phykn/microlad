import unittest

import torch

from src.pipeline.predict.guidance.metrics.conductance import (
    ConductanceSolver,
    compute_conductance,
)
from src.pipeline.predict.guidance.metrics.correlation import compute_correlation
from src.pipeline.predict.guidance.metrics.interface import (
    compute_interface_density,
)
from src.pipeline.predict.guidance.metrics.loss import (
    descriptor_loss,
    sample_descriptor_loss,
)


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

    def test_sample_descriptors_are_averaged_before_target_comparison(self):
        decoded = torch.stack((torch.zeros(2, 2), torch.ones(2, 2)))

        loss, _ = sample_descriptor_loss(
            decoded,
            num_phases=2,
            fraction_targets=torch.tensor([0.5, 0.5]),
            fraction_weight=1.0,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-6)

    def test_each_batch_descriptor_is_compared_after_averaging(self):
        decoded = torch.stack((torch.zeros(4, 4), torch.ones(4, 4)))
        solver = ConductanceSolver(4, 4, low_cond=0.1)
        cases = (
            {
                "tpc_targets": compute_correlation(
                    decoded,
                    num_phases=2,
                    temperature=0.01,
                ),
                "tpc_weight": 1.0,
            },
            {
                "sa_targets": compute_interface_density(
                    decoded,
                    num_phases=2,
                    temperature=0.01,
                    kernel_size=3,
                ),
                "sa_weight": 1.0,
                "sa_kernel_size": 3,
            },
            {
                "diffusivity_targets": compute_conductance(
                    decoded,
                    solver=solver,
                    num_phases=2,
                    temperature=0.01,
                ),
                "diffusivity_solver": solver,
                "diffusivity_weight": 1.0,
            },
        )

        for kwargs in cases:
            with self.subTest(descriptor=next(iter(kwargs))):
                loss, _ = sample_descriptor_loss(
                    decoded,
                    num_phases=2,
                    temperature=0.01,
                    **kwargs,
                )
                self.assertLess(float(loss), 1e-8)


if __name__ == "__main__":
    unittest.main()
