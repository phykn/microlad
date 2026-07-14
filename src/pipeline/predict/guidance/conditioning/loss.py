import torch
import torch.nn.functional as F

from src.modeling.phases.relaxation import as_phase_probability_batch
from src.modeling.phases.representation import phase_loss, phase_target_indices
from src.validation import require_finite


def anchor_loss(
    values: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    num_phases: int,
    temperature: float = 0.1,
    weight: float = 1.0,
    phase_probabilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if values.ndim != 2:
        raise ValueError("values must have shape [H, W].")

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    require_finite("values", values)

    target_image = _as_image(target, device=values.device, dtype=values.dtype)
    if target_image.shape != values.shape:
        raise ValueError("target shape must match values shape.")
    require_finite("target", target_image)

    active = None
    if mask is not None:
        mask_image = _as_image(mask, device=values.device, dtype=values.dtype)
        if mask_image.shape != values.shape:
            raise ValueError("mask shape must match values shape.")
        require_finite("mask", mask_image)
        active = mask_image > 0
        if not bool(active.any().item()):
            zero = values.sum() * 0.0
            return zero, {
                "anchor_mse": zero.detach(),
                "anchor_phase": zero.detach(),
            }

    target_batch = target_image.view(1, 1, *target_image.shape)
    indices = phase_target_indices(target_batch.round(), num_phases)

    if phase_probabilities is not None:
        probability = as_phase_probability_batch(
            phase_probabilities,
            num_phases=num_phases,
        )
        if probability.shape[0] != 1 or probability.shape[-2:] != values.shape:
            raise ValueError("phase probability shape must match values shape.")
        one_hot = F.one_hot(indices, num_classes=num_phases).movedim(-1, 1)
        if active is None:
            selected_probability = probability
            selected_one_hot = one_hot.to(probability.dtype)
            selected_indices = indices
        else:
            selected_probability = probability[0, :, active].transpose(0, 1)
            selected_one_hot = one_hot[0, :, active].transpose(0, 1).to(
                probability.dtype
            )
            selected_indices = indices[0, active]
        mse = F.mse_loss(selected_probability, selected_one_hot)
        phase = F.nll_loss(
            selected_probability.clamp_min(
                torch.finfo(selected_probability.dtype).tiny
            ).log(),
            selected_indices,
        )
    else:
        if active is None:
            selected_values = values.view(1, 1, *values.shape)
            selected_target = target_batch
        else:
            selected_values = values[active].view(1, 1, 1, -1)
            selected_target = target_image[active].view(1, 1, 1, -1)
        mse = F.mse_loss(selected_values, selected_target)
        phase = phase_loss(
            selected_values,
            selected_target.round(),
            num_phases,
            temperature,
        )

    return weight * (mse + phase), {
        "anchor_mse": mse.detach(),
        "anchor_phase": phase.detach(),
    }


def _as_image(
    target: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    target = target.to(device=device, dtype=dtype)

    if target.ndim == 2:
        return target

    if target.ndim == 4 and target.shape[:2] == (1, 1):
        return target[0, 0]

    raise ValueError("target must have shape [H, W] or [1, 1, H, W].")
