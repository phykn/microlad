import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.pipeline.predict.guidance.metrics.runs import (
    compute_run_profile,
    run_profile_loss,
)


def _one_hot(labels: torch.Tensor, num_phases: int = 2) -> torch.Tensor:
    return (
        F.one_hot(labels.to(torch.long), num_classes=num_phases)
        .movedim(-1, 1)
        .float()
    )


class RunProfileTest(unittest.TestCase):
    def test_checkerboard_has_no_length_two_run(self):
        image = torch.from_numpy(np.indices((4, 4)).sum(axis=0) % 2).unsqueeze(0)
        volume = torch.from_numpy(
            np.indices((4, 4, 4)).sum(axis=0) % 2
        ).unsqueeze(0)

        target = compute_run_profile(
            _one_hot(image),
            lengths=(2, 4),
        ).mean(dim=0)
        actual = compute_run_profile(
            _one_hot(volume),
            lengths=(2, 4),
        )

        self.assertEqual(actual.shape, torch.Size([3, 2, 2]))
        self.assertTrue(torch.allclose(actual, target.unsqueeze(0).expand_as(actual)))
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_uniform_phase_has_full_survival_for_occupied_phase(self):
        labels = torch.zeros(1, 4, 4, dtype=torch.long)

        profile = compute_run_profile(
            _one_hot(labels),
            lengths=(2, 4),
        )

        self.assertTrue(torch.equal(profile[:, 0], torch.ones_like(profile[:, 0])))
        self.assertTrue(torch.equal(profile[:, 1], torch.zeros_like(profile[:, 1])))

    def test_known_run_counts_match_window_products(self):
        labels = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]]
        )

        profile = compute_run_profile(
            _one_hot(labels),
            lengths=(2,),
        )

        expected = torch.tensor(
            [
                [[1.0], [1.0]],
                [[2.0 / 3.0], [2.0 / 3.0]],
            ]
        )
        self.assertTrue(torch.allclose(profile, expected))

    def test_profile_detects_long_runs_in_only_one_direction(self):
        labels = torch.arange(8).remainder(2).view(1, 8, 1).expand(1, 8, 8)

        profile = compute_run_profile(
            _one_hot(labels),
            lengths=(2, 4, 8),
        )

        self.assertTrue(torch.all(profile[1] > profile[0]))

    def test_loss_has_soft_gradients_when_profiles_differ(self):
        labels = torch.arange(4).remainder(2).view(1, 4, 1).expand(1, 4, 4)
        probabilities = _one_hot(labels).requires_grad_(True)
        target = torch.zeros(2, 2)

        loss, stats = run_profile_loss(
            probabilities,
            target,
            lengths=(2, 4),
        )
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(probabilities.grad.abs().sum()), 0.0)
        self.assertEqual(stats["actual_run_profile"].shape, torch.Size([2, 2, 2]))

    def test_product_keeps_gradient_at_a_single_zero_probability(self):
        probabilities = torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]],
            requires_grad=True,
        )

        profile = compute_run_profile(probabilities, lengths=(2,))
        profile[1, 0, 0].backward()

        self.assertGreater(float(probabilities.grad[0, 0, 0, 1]), 0.0)

    def test_profile_rejects_invalid_probabilities_and_lengths(self):
        probabilities = torch.full((1, 2, 4, 4), 0.5)

        with self.assertRaisesRegex(ValueError, "sorted"):
            compute_run_profile(probabilities, lengths=(4, 2))
        with self.assertRaisesRegex(ValueError, "exceed"):
            compute_run_profile(probabilities, lengths=(2, 8))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            compute_run_profile(probabilities * 0.5, lengths=(2, 4))


if __name__ == "__main__":
    unittest.main()
