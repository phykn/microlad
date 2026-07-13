import unittest

import torch

from src.pipelines.guidance.metrics.conductance import (
    ConductanceSolver,
    conductance_loss,
)


class ConductanceLossTest(unittest.TestCase):
    def test_solver_matches_uniform_material_conductance(self):
        solver = ConductanceSolver(height=4, width=4, low_cond=0.1)

        high = solver(torch.ones(4, 4))
        middle = solver(torch.full((4, 4), 0.5))
        low = solver(torch.zeros(4, 4))

        self.assertTrue(torch.allclose(high, torch.tensor(1.0), atol=1e-4))
        self.assertTrue(torch.allclose(low, torch.tensor(0.1), atol=1e-4))
        self.assertTrue(low < middle < high)

    def test_solver_clips_zero_low_cond_to_internal_floor(self):
        solver = ConductanceSolver(height=2, width=2, low_cond=0.0)

        self.assertEqual(solver.low_cond, 0.001)

    def test_conductance_loss_matches_phase_targets_with_real_solver(self):
        solver = ConductanceSolver(height=4, width=4, low_cond=0.1)
        values = torch.ones(4, 4)
        targets = torch.tensor([0.1, 1.0])

        loss, stats = conductance_loss(
            values,
            targets,
            solver=solver,
            num_phases=2,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(
            torch.allclose(stats["actual_conductance"], targets, atol=1e-4)
        )
        self.assertTrue(torch.allclose(stats["target_conductance"], targets))

    def test_conductance_loss_resizes_masks_to_solver_size(self):
        solver = ConductanceSolver(height=2, width=2, low_cond=0.1)
        values = torch.ones(4, 4)

        loss, stats = conductance_loss(
            values,
            {0: 0.1, 1: 1.0},
            solver=solver,
            num_phases=2,
            temperature=0.01,
        )

        self.assertLess(float(loss), 1e-4)
        self.assertTrue(
            torch.allclose(
                stats["actual_conductance"],
                torch.tensor([0.1, 1.0]),
                atol=1e-4,
            )
        )

    def test_conductance_loss_clamps_targets_to_solver_range(self):
        solver = ConductanceSolver(height=2, width=2, low_cond=0.1)
        values = torch.ones(2, 2)

        _, stats = conductance_loss(
            values,
            torch.tensor([0.0, 2.0]),
            solver=solver,
            num_phases=2,
        )

        self.assertTrue(
            torch.allclose(
                stats["target_conductance"],
                torch.tensor([0.1, 1.0]),
            )
        )

    def test_conductance_loss_is_differentiable_through_solver(self):
        solver = ConductanceSolver(height=2, width=2, low_cond=0.1)
        values = torch.tensor(
            [[-0.5, 0.5], [0.25, -0.25]],
            requires_grad=True,
        )

        loss, _ = conductance_loss(
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
        self.assertGreater(float(torch.linalg.vector_norm(values.grad)), 0.0)

    def test_conductance_loss_rejects_invalid_inputs(self):
        solver = ConductanceSolver(height=2, width=2)
        with self.assertRaisesRegex(ValueError, "values"):
            conductance_loss(
                torch.zeros(1),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "values"):
            conductance_loss(
                torch.empty(0, 4),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "values"):
            conductance_loss(
                torch.empty(4, 0),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "num_phases"):
            conductance_loss(
                torch.zeros(2, 2),
                torch.zeros(1),
                solver=solver,
                num_phases=1,
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            conductance_loss(
                torch.zeros(2, 2),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
                temperature=0.0,
            )
        with self.assertRaisesRegex(ValueError, "weight"):
            conductance_loss(
                torch.zeros(2, 2),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
                weight=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "targets"):
            conductance_loss(
                torch.zeros(2, 2),
                {0: 0.0},
                solver=solver,
                num_phases=2,
            )
        with self.assertRaisesRegex(ValueError, "targets"):
            conductance_loss(
                torch.zeros(2, 2),
                torch.zeros(3),
                solver=solver,
                num_phases=2,
            )

    def test_solver_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "height"):
            ConductanceSolver(height=2.5, width=2)
        with self.assertRaisesRegex(ValueError, "width"):
            ConductanceSolver(height=2, width=0)
        with self.assertRaisesRegex(ValueError, "low_cond"):
            ConductanceSolver(height=2, width=2, low_cond=float("nan"))

        solver = ConductanceSolver(height=2, width=2)
        with self.assertRaisesRegex(ValueError, "mask"):
            solver(torch.zeros(3, 3))

    def test_solver_rejects_non_finite_or_out_of_range_masks(self):
        solver = ConductanceSolver(height=2, width=2)

        with self.assertRaisesRegex(ValueError, "finite"):
            solver(torch.full((2, 2), float("nan")))

        for mask in (
            torch.full((2, 2), -0.1),
            torch.full((2, 2), 1.1),
        ):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    solver(mask)

    def test_conductance_loss_rejects_non_finite_values_and_targets(self):
        solver = ConductanceSolver(height=2, width=2)

        with self.assertRaisesRegex(ValueError, "finite"):
            conductance_loss(
                torch.full((2, 2), float("nan")),
                torch.zeros(2),
                solver=solver,
                num_phases=2,
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            conductance_loss(
                torch.zeros(2, 2),
                torch.tensor([0.1, float("inf")]),
                solver=solver,
                num_phases=2,
            )


if __name__ == "__main__":
    unittest.main()
