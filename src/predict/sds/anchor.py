import torch
import torch.nn.functional as F

from src.loss.phase import phase_loss


def anchor_loss(
    values: torch.Tensor,
    target: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if values.ndim != 2:
        raise ValueError("values must have shape [H, W].")
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    target_image = _as_image(target, device=values.device, dtype=values.dtype)
    if target_image.shape != values.shape:
        raise ValueError("target shape must match values shape.")

    recon = values.view(1, 1, *values.shape)
    target_batch = target_image.view(1, 1, *target_image.shape)
    mse = F.mse_loss(recon, target_batch)
    phase = phase_loss(recon, target_batch, num_phases, temperature)
    loss = weight * (mse + phase)
    return loss, {
        "anchor_mse": mse.detach(),
        "anchor_phase": phase.detach(),
    }


def masked_anchor_loss(
    values: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if values.ndim != 2:
        raise ValueError("values must have shape [H, W].")
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    target_image = _as_image(target, device=values.device, dtype=values.dtype)
    mask_image = _as_image(mask, device=values.device, dtype=values.dtype)
    if target_image.shape != values.shape:
        raise ValueError("target shape must match values shape.")
    if mask_image.shape != values.shape:
        raise ValueError("mask shape must match values shape.")

    active = mask_image > 0
    if not bool(active.any().item()):
        zero = values.sum() * 0.0
        return zero, {
            "anchor_mse": zero.detach(),
            "anchor_phase": zero.detach(),
        }

    selected_values = values[active].view(1, 1, 1, -1)
    selected_target = target_image[active].view(1, 1, 1, -1)
    mse = F.mse_loss(selected_values, selected_target)
    phase = phase_loss(selected_values, selected_target, num_phases, temperature)
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
