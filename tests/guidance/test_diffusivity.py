import unittest

import torch

from src.pipelines.guidance.physics.diffusivity import DiffusivitySolver, diffusivity_loss


class PredictSDSDiffusivityTest(unittest.TestCase):
    def test_solver_matches_uniform_material_diffusivity(self):
        solver = DiffusivitySolver(height=4, width=4, low_cond=0.1)

        high = solver(torch.ones(4, 4))
        low = solver(torch.zeros(4, 4))

        self.assertTrue(torch.allclose(high, torch.tensor(1.0), atol=1e-4))
        self.assertTrue(torch.allclose(low, torch.tensor(0.1), atol=1e-4))

    def test_solver_clips_zero_low_cond_to_internal_floor(self):
        solver = DiffusivitySolver(height=2, width=2, low_cond=0.0)

        self.assertEqual(solver.low_cond, 0.001)

    def test_diffusivity_loss_matches_phase_targets_with_real_solver(self):
        solver = DiffusivitySolver(height=4, width=4, low_cond=0.1)
        values = torch.ones(4, 4)
        targets = torch.tensor([0.1, 1.0])

        loss, stats = diffusivity_loss(
            values,
            targets,
            solver=solver,
            num_phases=2,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(torch.allclose(stats["actual_diffusivity"], targets, atol=1e-4))
        self.assertTrue(torch.allclose(stats["target_diffusivity"], targets))

    def test_diffusivity_loss_resizes_masks_to_solver_size(self):
        solver = DiffusivitySolver(height=2, width=2, low_cond=0.1)
        values = torch.ones(4, 4)

        loss, stats = diffusivity_loss(
            values,
            {0: 0.1, 1: 1.0},
            solver=solver,
            num_phases=2,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(
            torch.allclose(
                stats["actual_diffusivity"],
                torch.tensor([0.1, 1.0]),
                atol=1e-4,
            )
        )

    def test_diffusivity_loss_clamps_targets_to_solver_range(self):
        solver = DiffusivitySolver(height=2, width=2, low_cond=0.1)
        values = torch.ones(2, 2)

        _, stats = diffusivity_loss(
            values,
            torch.tensor([0.0, 2.0]),
            solver=solver,
            num_phases=2,
        )

        self.assertTrue(
            torch.allclose(
                stats["target_diffusivity"],
                torch.tensor([0.1, 1.0]),
            )
        )

    def test_diffusivity_loss_is_differentiable_through_solver(self):
        solver = DiffusivitySolver(height=2, width=2, low_cond=0.1)
        values = torch.tensor(
            [[-0.5, 0.5], [0.25, -0.25]],
            requires_grad=True,
        )

        loss, _ = diffusivity_loss(
            values,
            torch.zeros(2),
            solver=solver,
            num_phases=2,
            temperature=0.5,
        )
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertEqual(values.grad.shape, values.shape)
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_diffusivity_loss_rejects_invalid_inputs(self):
        solver = DiffusivitySolver(height=2, width=2)
        with self.assertRaisesRegex(ValueError, "values"):
            diffusivity_loss(torch.zeros(1), torch.zeros(2), solver=solver, num_phases=2)
        with self.assertRaisesRegex(ValueError, "values"):
            diffusivity_loss(
                torch.empty(0, 4),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "values"):
            diffusivity_loss(
                torch.empty(4, 0),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "num_phases"):
            diffusivity_loss(torch.zeros(2, 2), torch.zeros(1), solver=solver, num_phases=1)
        with self.assertRaisesRegex(ValueError, "temperature"):
            diffusivity_loss(
                torch.zeros(2, 2),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
                temperature=0.0,
            )
        with self.assertRaisesRegex(ValueError, "weight"):
            diffusivity_loss(
                torch.zeros(2, 2),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
                weight=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "targets"):
            diffusivity_loss(torch.zeros(2, 2), {0: 0.0}, solver=solver, num_phases=2)
        with self.assertRaisesRegex(ValueError, "targets"):
            diffusivity_loss(torch.zeros(2, 2), torch.zeros(3), solver=solver, num_phases=2)

    def test_solver_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "height"):
            DiffusivitySolver(height=0, width=2)
        with self.assertRaisesRegex(ValueError, "width"):
            DiffusivitySolver(height=2, width=0)
        with self.assertRaisesRegex(ValueError, "low_cond"):
            DiffusivitySolver(height=2, width=2, low_cond=-0.1)

        solver = DiffusivitySolver(height=2, width=2)
        with self.assertRaisesRegex(ValueError, "mask"):
            solver(torch.zeros(3, 3))

    def test_solver_rejects_non_finite_or_out_of_range_masks(self):
        solver = DiffusivitySolver(height=2, width=2)

        with self.assertRaisesRegex(ValueError, "finite"):
            solver(torch.full((2, 2), float("nan")))

        for mask in (
            torch.full((2, 2), -0.1),
            torch.full((2, 2), 1.1),
        ):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    solver(mask)

    def test_diffusivity_loss_rejects_non_finite_values_and_targets(self):
        solver = DiffusivitySolver(height=2, width=2)

        with self.assertRaisesRegex(ValueError, "finite"):
            diffusivity_loss(
                torch.full((2, 2), float("nan")),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            diffusivity_loss(
                torch.zeros(2, 2),
                torch.tensor([0.1, float("inf")]),
                solver=solver,
                num_phases=2,
            )


if __name__ == "__main__":
    unittest.main()
