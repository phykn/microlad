import unittest

import torch

from src.pipelines.guidance.metrics.correlation import (
    compute_correlation,
    correlation_loss,
)


class CorrelationLossTest(unittest.TestCase):
    def test_correlation_is_invariant_to_periodic_translation(self):
        values = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )
        shifted = torch.roll(values, shifts=(1, 2), dims=(0, 1))

        actual = compute_correlation(values, num_phases=2, temperature=0.01)
        translated = compute_correlation(shifted, num_phases=2, temperature=0.01)

        self.assertTrue(torch.allclose(actual, translated, atol=1e-6))

    def test_correlation_loss_matches_constant_phase_target(self):
        values = torch.full((4, 4), 0.0)
        targets = torch.zeros(2, 4)
        targets[0] = 1.0

        loss, stats = correlation_loss(
            values,
            targets,
            num_phases=2,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertEqual(stats["actual_correlation"].shape, torch.Size([2, 4]))
        self.assertTrue(torch.allclose(stats["target_correlation"], targets))

    def test_correlation_loss_accepts_phase_mapping_targets(self):
        values = torch.full((4, 4), 1.0)
        targets = {
            0: torch.zeros(4),
            1: torch.ones(4),
        }

        loss, stats = correlation_loss(
            values,
            targets,
            num_phases=2,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(
            torch.allclose(
                stats["actual_correlation"][1],
                torch.ones(4),
                atol=1e-3,
            )
        )

    def test_correlation_loss_is_differentiable(self):
        values = torch.tensor(
            [[[0.25, 0.75], [0.5, 0.1]]],
            requires_grad=True,
        )
        targets = torch.zeros(2, 2)

        loss, _ = correlation_loss(
            values,
            targets,
            num_phases=2,
            temperature=0.5,
        )
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertEqual(values.grad.shape, values.shape)

    def test_correlation_loss_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "values"):
            correlation_loss(torch.zeros(1), torch.zeros(2, 1), num_phases=2)
        with self.assertRaisesRegex(ValueError, "values"):
            correlation_loss(torch.empty(0, 4), torch.zeros(2, 1), num_phases=2)
        with self.assertRaisesRegex(ValueError, "values"):
            correlation_loss(torch.empty(4, 0), torch.zeros(2, 1), num_phases=2)
        with self.assertRaisesRegex(ValueError, "num_phases"):
            correlation_loss(torch.zeros(2, 2), torch.zeros(1, 1), num_phases=1)
        with self.assertRaisesRegex(ValueError, "temperature"):
            correlation_loss(
                torch.zeros(2, 2),
                torch.zeros(2, 1),
                num_phases=2,
                temperature=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "weight"):
            correlation_loss(
                torch.zeros(2, 2),
                torch.zeros(2, 1),
                num_phases=2,
                weight=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "targets"):
            correlation_loss(torch.zeros(2, 2), {0: torch.zeros(1)}, num_phases=2)
        with self.assertRaisesRegex(ValueError, "targets"):
            correlation_loss(torch.zeros(2, 2), torch.zeros(3, 1), num_phases=2)
        with self.assertRaisesRegex(ValueError, "length"):
            correlation_loss(torch.zeros(4, 4), torch.zeros(2, 3), num_phases=2)
        with self.assertRaisesRegex(ValueError, "phase indices"):
            correlation_loss(
                torch.zeros(2, 2),
                {0.5: torch.zeros(2), 1: torch.zeros(2)},
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            correlation_loss(
                torch.zeros(2, 2),
                torch.full((2, 2), float("nan")),
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            correlation_loss(
                torch.zeros(2, 2),
                torch.full((2, 2), -1.0),
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            correlation_loss(
                torch.zeros(2, 2),
                torch.full((2, 2), 1.1),
                num_phases=2,
            )


if __name__ == "__main__":
    unittest.main()
