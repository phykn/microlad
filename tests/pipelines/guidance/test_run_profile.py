import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.pipelines.guidance.descriptors.run_profile import (
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
    def test_checkerboard_has_the_same_profile_in_2d_and_every_3d_axis(self):
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
