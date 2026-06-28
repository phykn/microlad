import unittest

import torch

from src.predict.sds import anchor_loss


class PredictSDSAnchorTest(unittest.TestCase):
    def test_anchor_loss_is_zero_for_matching_images(self):
        values = torch.tensor([[-1.0, 1.0], [0.0, 0.0]])
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


if __name__ == "__main__":
    unittest.main()
