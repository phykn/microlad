import torch
import torch.nn.functional as F


def _validate_num_phases(num_phases: int) -> None:
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")


def phase_levels(
    num_phases: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    _validate_num_phases(num_phases)

    return torch.arange(num_phases, device=device, dtype=dtype)


def phase_logits(
    recon: torch.Tensor,
    num_phases: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    _validate_num_phases(num_phases)

    if recon.ndim != 4 or recon.shape[1] != 1:
        raise ValueError("recon must have shape [B, 1, H, W].")

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    levels = phase_levels(num_phases, device=recon.device, dtype=recon.dtype)
    distance = recon - levels.view(1, num_phases, 1, 1)
    return -(distance.pow(2)) / temperature


def phase_target_indices(target: torch.Tensor, num_phases: int) -> torch.Tensor:
    _validate_num_phases(num_phases)

    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("target must have shape [B, 1, H, W].")

    if any(size <= 0 for size in target.shape):
        raise ValueError("target must not be empty.")

    if not torch.isfinite(target).all():
        raise ValueError("target phase values must be finite.")

    rounded = target.round()
    if not torch.equal(target, rounded):
        raise ValueError("target phase values must be integer values.")

    if target.min().item() < 0 or target.max().item() >= num_phases:
        raise ValueError(f"target phase values must be from 0 to {num_phases - 1}.")

    return rounded.to(torch.long).squeeze(1)


def phase_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    _validate_num_phases(num_phases)

    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, num_phases, H, W].")

    if logits.shape[1] != num_phases:
        raise ValueError("logits channel count must match num_phases.")

    if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError("recon and target spatial shape must match.")

    if any(size <= 0 for size in logits.shape):
        raise ValueError("recon and target must not be empty.")

    return F.cross_entropy(logits, phase_target_indices(target, num_phases))


def logits_to_phase_values(logits: torch.Tensor, num_phases: int) -> torch.Tensor:
    _validate_num_phases(num_phases)

    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, num_phases, H, W].")

    if logits.shape[1] != num_phases:
        raise ValueError("logits channel count must match num_phases.")

    if any(size <= 0 for size in logits.shape):
        raise ValueError("logits must not be empty.")

    levels = phase_levels(num_phases, device=logits.device, dtype=logits.dtype)
    probability = torch.softmax(logits, dim=1)
    return (probability * levels.view(1, num_phases, 1, 1)).sum(
        dim=1,
        keepdim=True,
    )


def phase_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    num_phases: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    if recon.shape != target.shape:
        raise ValueError("recon and target must have the same shape.")

    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("target must have shape [B, 1, H, W].")

    if any(size <= 0 for size in target.shape):
        raise ValueError("recon and target must not be empty.")

    logits = phase_logits(recon, num_phases, temperature)
    return F.cross_entropy(logits, phase_target_indices(target, num_phases))
