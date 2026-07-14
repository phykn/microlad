import unittest

import torch

from src.pipeline.predict.guidance.metrics.fraction import phase_fraction_loss


class PhaseFractionLossTest(unittest.TestCase):
    def test_categorical_probabilities_do_not_collapse_into_middle_phase(self):
        values = torch.ones(2, 2)
        probabilities = torch.zeros(3, 2, 2)
        probabilities[0] = 0.5
        probabilities[2] = 0.5

        loss, stats = phase_fraction_loss(
            values,
            torch.tensor([0.5, 0.0, 0.5]),
            num_phases=3,
            phase_probabilities=probabilities,
        )

        self.assertLess(float(loss), 1e-8)
        self.assertTrue(
            torch.allclose(
                stats["actual_fraction"],
                torch.tensor([0.5, 0.0, 0.5]),
            )
        )

    def test_phase_fraction_loss_matches_each_phase_fraction_directly(self):
        values = torch.tensor([[0.0, 1.0, 2.0]])
        targets = {0: 1.0 / 3.0, 1: 1.0 / 3.0, 2: 1.0 / 3.0}

        loss, stats = phase_fraction_loss(
            values,
            targets,
            num_phases=3,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(
            torch.allclose(
                stats["actual_fraction"],
                torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
                atol=1e-3,
            )
        )
        self.assertTrue(
            torch.allclose(
                stats["target_fraction"],
                torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            )
        )

    def test_phase_fraction_loss_handles_more_than_three_phases(self):
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        targets = torch.tensor([0.25, 0.25, 0.25, 0.25])

        loss, stats = phase_fraction_loss(
            values,
            targets,
            num_phases=4,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(
            torch.allclose(
                stats["actual_fraction"],
                torch.tensor([0.25, 0.25, 0.25, 0.25]),
                atol=1e-3,
            )
        )

    def test_phase_fraction_loss_is_differentiable(self):
        values = torch.tensor([[[0.25, 0.75], [0.5, 0.1]]], requires_grad=True)

        loss, _ = phase_fraction_loss(
            values,
            {0: 0.5, 1: 0.5},
            num_phases=2,
        )
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertEqual(values.grad.shape, values.shape)

    def test_phase_fraction_loss_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "values"):
            phase_fraction_loss(torch.empty(0), {0: 0.5, 1: 0.5}, num_phases=2)
        with self.assertRaisesRegex(ValueError, "num_phases"):
            phase_fraction_loss(torch.zeros(1), {0: 1.0}, num_phases=1)
        with self.assertRaisesRegex(ValueError, "targets"):
            phase_fraction_loss(torch.zeros(1), {0: 1.0}, num_phases=2)
        with self.assertRaisesRegex(ValueError, "sum"):
            phase_fraction_loss(torch.zeros(1), {0: 0.2, 1: 0.2}, num_phases=2)
        with self.assertRaisesRegex(ValueError, "temperature"):
            phase_fraction_loss(
                torch.zeros(1),
                {0: 0.5, 1: 0.5},
                num_phases=2,
                temperature=0.0,
            )
        with self.assertRaisesRegex(ValueError, "weight"):
            phase_fraction_loss(
                torch.zeros(1),
                {0: 0.5, 1: 0.5},
                num_phases=2,
                weight=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
