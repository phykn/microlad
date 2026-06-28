import unittest

import torch

from src.predict.sds.diffusivity import DiffusivitySolver
from src.predict.sds.objective import descriptor_loss


class PredictSDSObjectiveTest(unittest.TestCase):
    def test_descriptor_loss_rejects_negative_weights(self):
        decoded = torch.zeros(2, 2)
        solver = DiffusivitySolver(height=2, width=2)

        for kwargs in (
            {"vf_targets": torch.tensor([0.5, 0.5]), "vf_weight": -1.0},
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
            {"vf_weight": 1.0},
            {"tpc_weight": 1.0},
            {"sa_weight": 1.0},
            {"diffusivity_weight": 1.0, "diffusivity_solver": DiffusivitySolver(2, 2)},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "targets"):
                    descriptor_loss(decoded, num_phases=2, **kwargs)

    def test_descriptor_loss_allows_missing_targets_when_unweighted(self):
        decoded = torch.ones(2, 2)

        loss, stats = descriptor_loss(decoded, num_phases=2)

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))
        self.assertEqual(stats, {})


if __name__ == "__main__":
    unittest.main()
