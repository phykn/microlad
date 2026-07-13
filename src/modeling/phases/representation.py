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


def _validate_phase_logits(logits: torch.Tensor, num_phases: int) -> None:
    _validate_num_phases(num_phases)

    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, num_phases, H, W].")

    if logits.shape[1] != num_phases:
        raise ValueError("logits channel count must match num_phases.")

    if any(size <= 0 for size in logits.shape):
        raise ValueError("logits must not be empty.")


def logits_to_probabilities(
    logits: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    _validate_phase_logits(logits, num_phases)
    return torch.softmax(logits, dim=1)


def logits_to_relaxed_labels(
    logits: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    probabilities = logits_to_probabilities(logits, num_phases)

    levels = phase_levels(num_phases, device=logits.device, dtype=logits.dtype)
    return (probabilities * levels.view(1, num_phases, 1, 1)).sum(
        dim=1,
        keepdim=True,
    )


def probabilities_to_relaxed_labels(
    probabilities: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    _validate_phase_logits(probabilities, num_phases)

    levels = phase_levels(
        num_phases,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return (probabilities * levels.view(1, num_phases, 1, 1)).sum(
        dim=1,
        keepdim=True,
    )


def probabilities_to_labels(
    probabilities: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    _validate_phase_logits(probabilities, num_phases)
    return probabilities.argmax(dim=1, keepdim=True)


def geometric_probability_consensus(
    axis_probabilities: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    """Fuse axis-wise categorical predictions by normalized geometric mean."""
    _validate_num_phases(num_phases)

    if axis_probabilities.ndim < 3:
        raise ValueError(
            "axis_probabilities must have shape [num_axes, num_phases, ...spatial]."
        )
    if axis_probabilities.shape[0] <= 0:
        raise ValueError("axis_probabilities must contain at least one axis.")
    if axis_probabilities.shape[1] != num_phases:
        raise ValueError("probability channels must match num_phases.")
    if any(size <= 0 for size in axis_probabilities.shape):
        raise ValueError("axis_probabilities must not be empty.")
    if not axis_probabilities.is_floating_point():
        raise ValueError("axis_probabilities must be floating point.")
    if not torch.isfinite(axis_probabilities).all():
        raise ValueError("axis_probabilities must be finite.")
    if torch.any(axis_probabilities < 0.0) or torch.any(axis_probabilities > 1.0):
        raise ValueError("axis_probabilities must be between 0 and 1.")
    if not torch.allclose(
        axis_probabilities.sum(dim=1),
        torch.ones_like(axis_probabilities[:, 0]),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("axis_probabilities must sum to one across phases.")

    tiny = torch.finfo(axis_probabilities.dtype).tiny
    consensus_logits = axis_probabilities.clamp_min(tiny).log().mean(dim=0)
    return torch.softmax(consensus_logits, dim=0)


def logits_to_labels(logits: torch.Tensor, num_phases: int) -> torch.Tensor:
    _validate_phase_logits(logits, num_phases)
    return logits.argmax(dim=1, keepdim=True)


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
