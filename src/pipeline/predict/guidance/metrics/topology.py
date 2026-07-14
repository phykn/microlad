import torch

from src.validation import require_finite, require_finite_number


def compute_euler_density(
    probabilities: torch.Tensor,
    *,
    scale: float = 1000.0,
) -> torch.Tensor:
    _validate_probabilities(probabilities)
    require_finite_number("scale", scale)
    if scale <= 0.0:
        raise ValueError("scale must be positive.")

    pixels = probabilities.sum(dim=(2, 3))
    horizontal_edges = (
        probabilities[:, :, :, :-1] * probabilities[:, :, :, 1:]
    ).sum(dim=(2, 3))
    vertical_edges = (
        probabilities[:, :, :-1, :] * probabilities[:, :, 1:, :]
    ).sum(dim=(2, 3))
    filled_vertices = (
        probabilities[:, :, :-1, :-1]
        * probabilities[:, :, 1:, :-1]
        * probabilities[:, :, :-1, 1:]
        * probabilities[:, :, 1:, 1:]
    ).sum(dim=(2, 3))
    characteristic = (
        pixels - horizontal_edges - vertical_edges + filled_vertices
    )
    area = probabilities.shape[2] * probabilities.shape[3]
    return characteristic * (float(scale) / area)


def _validate_probabilities(probabilities: torch.Tensor) -> None:
    if probabilities.ndim != 4:
        raise ValueError("probabilities must have shape [B, P, H, W].")
    if any(size <= 0 for size in probabilities.shape):
        raise ValueError("probabilities must not be empty.")
    if probabilities.shape[1] < 2:
        raise ValueError("probabilities must contain at least two phases.")
    if min(probabilities.shape[2:]) < 2:
        raise ValueError("probability images must be at least 2 by 2.")
    if not probabilities.is_floating_point():
        raise ValueError("probabilities must be floating point.")
    require_finite("probabilities", probabilities)
    if torch.any(probabilities < 0.0) or torch.any(probabilities > 1.0):
        raise ValueError("probabilities must be between 0 and 1.")
    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("probabilities must sum to one across phases.")
