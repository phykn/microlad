import unittest

import torch

from src.modeling.phases.representation import geometric_probability_consensus


class PhaseRepresentationTest(unittest.TestCase):
    def test_geometric_consensus_prefers_phase_supported_by_two_axes(self):
        axis_probabilities = torch.tensor(
            [
                [[[1.0]], [[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]], [[1.0]]],
                [[[1.0]], [[0.0]], [[0.0]]],
            ]
        )

        consensus = geometric_probability_consensus(axis_probabilities, 3)

        self.assertEqual(consensus.shape, torch.Size([3, 1, 1]))
        self.assertGreater(float(consensus[0, 0, 0]), 0.99)
        self.assertTrue(torch.allclose(consensus.sum(dim=0), torch.ones(1, 1)))


if __name__ == "__main__":
    unittest.main()
