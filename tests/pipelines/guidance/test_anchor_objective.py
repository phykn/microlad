import unittest

import torch

from src.pipelines.guidance.anchor_objective import anchor_loss
from src.pipelines.guidance.anchor_objective import masked_anchor_loss


class PredictSDSAnchorTest(unittest.TestCase):
    def test_categorical_anchor_loss_uses_probabilities_not_expected_label(self):
        values = torch.ones(2, 2)
        target = torch.zeros(2, 2)
        probabilities = torch.zeros(3, 2, 2)
        probabilities[0] = 0.9
        probabilities[2] = 0.1

        scalar_loss, _ = anchor_loss(values, target, num_phases=3)
        categorical_loss, _ = anchor_loss(
            values,
            target,
            num_phases=3,
            phase_probabilities=probabilities,
        )

        self.assertLess(float(categorical_loss), float(scalar_loss))

    def test_anchor_loss_is_zero_for_matching_images(self):
        values = torch.tensor([[0.0, 2.0], [1.0, 1.0]])
        target = values.view(1, 1, 2, 2)

        loss, stats = anchor_loss(values, target, num_phases=3, temperature=0.01)

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(torch.allclose(stats["anchor_mse"], torch.tensor(0.0)))
        self.assertLess(float(stats["anchor_phase"]), 1e-4)

    def test_anchor_loss_is_differentiable(self):
        values = torch.zeros(2, 2, requires_grad=True)
        target = torch.ones(2, 2)

        loss, _ = anchor_loss(values, target, num_phases=2)
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertEqual(values.grad.shape, values.shape)
        self.assertTrue(torch.all(values.grad < 0.0))

    def test_anchor_loss_accepts_relaxed_reconstructed_target(self):
        values = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        target = torch.tensor([[0.25, 1.25], [1.25, 0.25]])

        loss, stats = anchor_loss(values, target, num_phases=2)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(stats["anchor_phase"]))
        self.assertIsNotNone(values.grad)

    def test_anchor_loss_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "values"):
            anchor_loss(torch.zeros(1), torch.zeros(1), num_phases=2)
        with self.assertRaisesRegex(ValueError, "target"):
            anchor_loss(torch.zeros(2, 2), torch.zeros(1, 2, 2), num_phases=2)
        with self.assertRaisesRegex(ValueError, "shape"):
            anchor_loss(torch.zeros(2, 2), torch.zeros(3, 3), num_phases=2)
        with self.assertRaisesRegex(ValueError, "num_phases"):
            anchor_loss(torch.zeros(2, 2), torch.zeros(2, 2), num_phases=1)
        with self.assertRaisesRegex(ValueError, "temperature"):
            anchor_loss(
                torch.zeros(2, 2),
                torch.zeros(2, 2),
                num_phases=2,
                temperature=0.0,
            )
        with self.assertRaisesRegex(ValueError, "weight"):
            anchor_loss(torch.zeros(2, 2), torch.zeros(2, 2), num_phases=2, weight=-1.0)

    def test_anchor_losses_reject_non_finite_inputs(self):
        valid = torch.zeros(2, 2)

        cases = [
            lambda: anchor_loss(
                torch.full((2, 2), float("inf")),
                valid,
                num_phases=2,
            ),
            lambda: anchor_loss(
                valid,
                torch.full((2, 2), float("nan")),
                num_phases=2,
            ),
            lambda: masked_anchor_loss(
                valid,
                valid,
                torch.full((2, 2), float("nan")),
                num_phases=2,
            ),
        ]

        for call in cases:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "finite"):
                    call()


if __name__ == "__main__":
    unittest.main()
