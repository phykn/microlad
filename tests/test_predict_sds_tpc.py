import unittest

import torch

from src.predict.sds import tpc_loss


class PredictSDSTPCTest(unittest.TestCase):
    def test_tpc_loss_matches_constant_phase_target(self):
        values = torch.full((4, 4), -1.0)
        targets = torch.zeros(2, 4)
        targets[0] = 1.0

        loss, stats = tpc_loss(
            values,
            targets,
            num_phases=2,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertEqual(stats["actual_tpc"].shape, torch.Size([2, 4]))
        self.assertTrue(torch.allclose(stats["target_tpc"], targets))

    def test_tpc_loss_accepts_phase_mapping_targets(self):
        values = torch.full((4, 4), 1.0)
        targets = {
            0: torch.zeros(4),
            1: torch.ones(4),
        }

        loss, stats = tpc_loss(
            values,
            targets,
            num_phases=2,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(torch.allclose(stats["actual_tpc"][1], torch.ones(4), atol=1e-3))

    def test_tpc_loss_is_differentiable(self):
        values = torch.tensor(
            [[[-0.5, 0.5], [0.25, -0.25]]],
            requires_grad=True,
        )
        targets = torch.zeros(2, 2)

        loss, _ = tpc_loss(
            values,
            targets,
            num_phases=2,
            temperature=0.5,
        )
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertEqual(values.grad.shape, values.shape)

    def test_tpc_loss_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "values"):
            tpc_loss(torch.zeros(1), torch.zeros(2, 1), num_phases=2)
        with self.assertRaisesRegex(ValueError, "num_phases"):
            tpc_loss(torch.zeros(2, 2), torch.zeros(1, 1), num_phases=1)
        with self.assertRaisesRegex(ValueError, "temperature"):
            tpc_loss(
                torch.zeros(2, 2),
                torch.zeros(2, 1),
                num_phases=2,
                temperature=0.0,
            )
        with self.assertRaisesRegex(ValueError, "weight"):
            tpc_loss(torch.zeros(2, 2), torch.zeros(2, 1), num_phases=2, weight=-1.0)
        with self.assertRaisesRegex(ValueError, "targets"):
            tpc_loss(torch.zeros(2, 2), {0: torch.zeros(1)}, num_phases=2)
        with self.assertRaisesRegex(ValueError, "targets"):
            tpc_loss(torch.zeros(2, 2), torch.zeros(3, 1), num_phases=2)
        with self.assertRaisesRegex(ValueError, "length"):
            tpc_loss(torch.zeros(4, 4), torch.zeros(2, 3), num_phases=2)


if __name__ == "__main__":
    unittest.main()
