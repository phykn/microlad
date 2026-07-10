import torch

from src.modeling.phases.representation import phase_levels
from src.common.tensors.validation import require_finite


def calc_phase_probs(
    values: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    phase_dim: int = 0,
) -> torch.Tensor:
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    if phase_dim < 0 or phase_dim > values.ndim:
        raise ValueError("phase_dim must be between 0 and values.ndim.")

    if not values.is_floating_point():
        raise ValueError("values must be a floating point tensor.")

    require_finite("values", values)

    levels = phase_levels(num_phases, device=values.device, dtype=values.dtype)
    level_shape = [1] * (values.ndim + 1)
    level_shape[phase_dim] = num_phases

    distance = values.unsqueeze(phase_dim) - levels.view(level_shape)

    return torch.softmax(-distance.pow(2) / temperature, dim=phase_dim)
