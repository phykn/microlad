import unittest

import torch

from src.modeling.slicegan import critic_loss, generator_loss, gradient_penalty


class SliceGANObjectiveTest(unittest.TestCase):
    def test_critic_loss_matches_wgan_gp_objective(self):
        loss = critic_loss(
            torch.tensor([2.0, 4.0]),
            torch.tensor([1.0, 3.0]),
            torch.tensor(0.5),
            gradient_weight=10.0,
        )

        self.assertEqual(float(loss), 4.0)

    def test_generator_loss_maximizes_fake_score(self):
        loss = generator_loss(torch.tensor([1.0, 3.0]))

        self.assertEqual(float(loss), -2.0)

    def test_gradient_penalty_uses_interpolated_critic_gradient(self):
        critic = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4, 1, bias=False),
        )
        with torch.no_grad():
            critic[1].weight.fill_(0.5)

        penalty = gradient_penalty(
            critic,
            torch.zeros(2, 1, 2, 2),
            torch.ones(2, 1, 2, 2),
        )

        self.assertTrue(torch.allclose(penalty, torch.zeros(())))

    def test_gradient_penalty_requires_matching_batches(self):
        critic = torch.nn.Identity()

        with self.assertRaisesRegex(ValueError, "same shape"):
            gradient_penalty(
                critic,
                torch.zeros(2, 1, 2, 2),
                torch.zeros(1, 1, 2, 2),
            )


if __name__ == "__main__":
    unittest.main()
