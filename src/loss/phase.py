import torch
import torch.nn.functional as F


def phase_levels(
    num_phases: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    return torch.linspace(-1.0, 1.0, num_phases, device=device, dtype=dtype)


def phase_logits(
    recon: torch.Tensor,
    num_phases: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    if recon.ndim != 4 or recon.shape[1] != 1:
        raise ValueError("recon must have shape [B, 1, H, W].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    levels = phase_levels(num_phases, device=recon.device, dtype=recon.dtype)
    distance = recon - levels.view(1, num_phases, 1, 1)
    return -(distance.pow(2)) / temperature


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

    logits = phase_logits(recon, num_phases, temperature)
    levels = phase_levels(num_phases, device=target.device, dtype=target.dtype)
    target_index = (target - levels.view(1, num_phases, 1, 1)).abs().argmin(dim=1)
    return F.cross_entropy(logits, target_index)
