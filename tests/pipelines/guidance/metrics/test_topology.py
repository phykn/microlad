import unittest

import torch
import torch.nn.functional as F

from src.pipelines.guidance.metrics.topology import (
    compute_euler_density,
)


def _binary_probabilities(mask: torch.Tensor) -> torch.Tensor:
    labels = mask.to(torch.long).unsqueeze(0)
    return F.one_hot(labels, num_classes=2).movedim(-1, 1).float()


class TopologyDescriptorTest(unittest.TestCase):
    def test_euler_density_distinguishes_a_filled_component_from_a_ring(self):
        filled = torch.ones(3, 3)
        ring = filled.clone()
        ring[1, 1] = 0.0

        filled_density = compute_euler_density(
            _binary_probabilities(filled),
            scale=9.0,
        )
        ring_density = compute_euler_density(
            _binary_probabilities(ring),
            scale=9.0,
        )

        self.assertAlmostEqual(float(filled_density[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(ring_density[0, 1]), 0.0, places=6)

    def test_euler_density_rejects_invalid_probabilities(self):
        probabilities = torch.full((1, 2, 3, 3), 0.5)

        with self.assertRaisesRegex(ValueError, "sum to one"):
            compute_euler_density(probabilities * 0.5)
        with self.assertRaisesRegex(ValueError, "positive"):
            compute_euler_density(probabilities, scale=0.0)


if __name__ == "__main__":
    unittest.main()
