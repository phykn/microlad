import torch

from src.predict.sds.diffusivity import DiffusivitySolver
from src.predict.sds.sa import compute_surface_area
from src.predict.sds.tpc import compute_tpc
from src.predict.sds.vf import compute_volume_fraction


def test_volume_fraction_sums_to_one():
    values = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]],
        dtype=torch.float64,
    )

    actual = compute_volume_fraction(
        values,
        num_phases=2,
        temperature=0.01,
    )

    assert torch.allclose(
        actual.sum(),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1e-8,
    )
    assert torch.allclose(
        actual,
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        atol=1e-8,
    )


def test_tpc_is_invariant_to_periodic_translation():
    values = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    shifted = torch.roll(values, shifts=(1, 2), dims=(0, 1))

    left = compute_tpc(values, num_phases=2, temperature=0.01)
    right = compute_tpc(shifted, num_phases=2, temperature=0.01)

    assert torch.allclose(left, right, atol=1e-6)


def test_surface_area_is_zero_for_homogeneous_phase_limit():
    values = torch.zeros(16, 16, dtype=torch.float64)

    actual = compute_surface_area(
        values,
        num_phases=2,
        temperature=0.01,
        kernel_size=3,
        sigma=1.0,
    )

    assert torch.all(actual < 1e-8)


def test_diffusivity_normalizes_uniform_conductor_to_one():
    solver = DiffusivitySolver(4, 4, low_cond=0.001)

    actual = solver(torch.ones(4, 4))

    assert torch.allclose(actual, torch.tensor(1.0), atol=1e-6)


def test_diffusivity_is_bounded_and_monotone_for_uniform_fields():
    solver = DiffusivitySolver(4, 4, low_cond=0.01)

    low = solver(torch.zeros(4, 4))
    middle = solver(torch.full((4, 4), 0.5))
    high = solver(torch.ones(4, 4))

    assert 0.0 < low < middle < high
    assert torch.allclose(high, torch.tensor(1.0), atol=1e-6)


def test_fem_gradient_is_finite_and_nonzero():
    values = torch.tensor(
        [[0.1, 0.9], [0.2, 0.8]],
        requires_grad=True,
    )
    solver = DiffusivitySolver(2, 2, low_cond=0.01)

    loss = solver(values)
    (gradient,) = torch.autograd.grad(loss, values)

    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient) > 0
