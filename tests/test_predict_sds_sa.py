import unittest

import torch

from src.predict.sds import compute_surface_area, surface_area_loss


class PredictSDSSurfaceAreaTest(unittest.TestCase):
    def test_surface_area_loss_matches_simple_split_target(self):
        values = torch.tensor(
            [
                [-1.0, -1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0, 1.0],
            ]
        )
        targets = torch.tensor([0.25, 0.25])

        loss, stats = surface_area_loss(
            values,
            targets,
            num_phases=2,
            temperature=0.01,
            kernel_size=1,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(torch.allclose(stats["actual_sa"], targets, atol=1e-3))
        self.assertTrue(torch.allclose(stats["target_sa"], targets))

    def test_surface_area_is_zero_for_uniform_phase_with_default_smoothing(self):
        values = torch.full((4, 4), -1.0)

        actual = compute_surface_area(values, num_phases=2, temperature=0.01)

        self.assertTrue(torch.allclose(actual, torch.zeros(2), atol=1e-6))

    def test_surface_area_loss_accepts_phase_mapping_targets(self):
        values = torch.full((4, 4), -1.0)
        targets = {0: 0.0, 1: 0.0}

        loss, stats = surface_area_loss(
            values,
            targets,
            num_phases=2,
            temperature=0.01,
            kernel_size=1,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(torch.allclose(stats["actual_sa"], torch.zeros(2), atol=1e-3))

    def test_surface_area_loss_is_differentiable(self):
        values = torch.tensor(
            [[[-0.5, 0.5], [0.25, -0.25]]],
            requires_grad=True,
        )

        loss, _ = surface_area_loss(
            values,
            torch.zeros(2),
            num_phases=2,
            temperature=0.5,
            kernel_size=1,
        )
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertEqual(values.grad.shape, values.shape)

    def test_surface_area_loss_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "values"):
            surface_area_loss(torch.zeros(1), torch.zeros(2), num_phases=2)
        with self.assertRaisesRegex(ValueError, "values"):
            surface_area_loss(torch.empty(0, 4), torch.zeros(2), num_phases=2)
        with self.assertRaisesRegex(ValueError, "values"):
            surface_area_loss(torch.empty(4, 0), torch.zeros(2), num_phases=2)
        with self.assertRaisesRegex(ValueError, "num_phases"):
            surface_area_loss(torch.zeros(2, 2), torch.zeros(1), num_phases=1)
        with self.assertRaisesRegex(ValueError, "temperature"):
            surface_area_loss(
                torch.zeros(2, 2),
                torch.zeros(2),
                num_phases=2,
                temperature=0.0,
            )
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            surface_area_loss(
                torch.zeros(2, 2),
                torch.zeros(2),
                num_phases=2,
                kernel_size=2,
            )
        with self.assertRaisesRegex(ValueError, "sigma"):
            surface_area_loss(
                torch.zeros(2, 2),
                torch.zeros(2),
                num_phases=2,
                sigma=0.0,
            )
        with self.assertRaisesRegex(ValueError, "weight"):
            surface_area_loss(
                torch.zeros(2, 2),
                torch.zeros(2),
                num_phases=2,
                weight=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "targets"):
            surface_area_loss(torch.zeros(2, 2), {0: 0.0}, num_phases=2)
        with self.assertRaisesRegex(ValueError, "targets"):
            surface_area_loss(torch.zeros(2, 2), torch.zeros(3), num_phases=2)
        with self.assertRaisesRegex(ValueError, "finite"):
            surface_area_loss(
                torch.zeros(2, 2),
                torch.tensor([float("nan"), 0.0]),
                num_phases=2,
            )


if __name__ == "__main__":
    unittest.main()
