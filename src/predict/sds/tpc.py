from collections.abc import Mapping

import torch
import torch.nn.functional as F

from src.predict.sds.phase import soft_phase_probability


def tpc_loss(
    values: torch.Tensor,
    targets: Mapping[int, torch.Tensor] | torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
    weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if values.ndim < 2:
        raise ValueError("values must have at least two spatial dimensions.")
    if values.numel() == 0 or values.shape[-2] <= 0 or values.shape[-1] <= 0:
        raise ValueError("values must have non-empty spatial dimensions.")
    if values.ndim == 4 and values.shape[1] != 1:
        raise ValueError("values with 4 dimensions must have shape [B, 1, H, W].")
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if weight < 0.0:
        raise ValueError("weight must be non-negative.")

    actual_tpc = compute_tpc(
        values,
        num_phases=num_phases,
        temperature=temperature,
    )
    target_tpc = _target_tensor(
        targets,
        num_phases=num_phases,
        device=values.device,
        dtype=values.dtype,
    )

    if target_tpc.shape[1] != actual_tpc.shape[1]:
        raise ValueError("targets length must match the predicted TPC length.")
    loss = weight * F.mse_loss(actual_tpc, target_tpc)

    stats = {
        "actual_tpc": actual_tpc.detach(),
        "target_tpc": target_tpc.detach(),
    }
    return loss, stats


def compute_tpc(
    values: torch.Tensor,
    *,
    num_phases: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("values must have at least two spatial dimensions.")
    if values.numel() == 0 or values.shape[-2] <= 0 or values.shape[-1] <= 0:
        raise ValueError("values must have non-empty spatial dimensions.")
    if values.ndim == 4 and values.shape[1] != 1:
        raise ValueError("values with 4 dimensions must have shape [B, 1, H, W].")
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    height, width = values.shape[-2:]
    slices = values.reshape(-1, height, width)
    bin_matrix, bin_counts = _build_tpc_bins(
        height,
        width,
        device=values.device,
        dtype=values.dtype,
    )

    probability = soft_phase_probability(
        slices,
        num_phases=num_phases,
        temperature=temperature,
        phase_dim=1,
    )
    return _phase_tpc(probability, bin_matrix, bin_counts)


def _phase_tpc(
    probability: torch.Tensor,
    bin_matrix: torch.Tensor,
    bin_counts: torch.Tensor,
) -> torch.Tensor:
    phase_profiles = []
    for phase in range(probability.shape[1]):
        tpc = _compute_tpc_batch(probability[:, phase], bin_matrix, bin_counts)
        phase_profiles.append(tpc.mean(dim=0))
    return torch.stack(phase_profiles, dim=0)


def _compute_tpc_batch(
    masks: torch.Tensor,
    bin_matrix: torch.Tensor,
    bin_counts: torch.Tensor,
) -> torch.Tensor:
    height, width = masks.shape[-2:]
    fft = torch.fft.fft2(masks, dim=(-2, -1))
    corr = torch.fft.ifft2(fft * torch.conj(fft), dim=(-2, -1)) / (height * width)
    corr = torch.real(torch.fft.fftshift(corr, dim=(-2, -1))).reshape(masks.shape[0], -1)
    return (corr @ bin_matrix.transpose(0, 1)) / bin_counts.squeeze(1).clamp_min(1)


def _build_tpc_bins(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if height <= 0 or width <= 0:
        raise ValueError("values must have non-empty spatial dimensions.")

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    radius = (
        ((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
        .sqrt()
        .round()
        .long()
        .view(-1)
    )
    num_bins = int(radius.max().item()) + 1
    bin_matrix = F.one_hot(radius, num_classes=num_bins).to(dtype=dtype).transpose(0, 1)
    bin_counts = bin_matrix.sum(dim=1, keepdim=True)
    return bin_matrix, bin_counts


def _target_tensor(
    targets: Mapping[int, torch.Tensor] | torch.Tensor,
    *,
    num_phases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        target = targets.to(device=device, dtype=dtype)
    else:
        expected_keys = set(range(num_phases))
        if set(int(phase) for phase in targets.keys()) != expected_keys:
            raise ValueError("targets must contain one TPC profile per phase.")
        target = torch.stack(
            [
                torch.as_tensor(targets[phase], device=device, dtype=dtype)
                for phase in range(num_phases)
            ],
            dim=0,
        )

    if target.ndim != 2 or target.shape[0] != num_phases:
        raise ValueError("targets must have shape [num_phases, num_bins].")
    if target.shape[1] < 1:
        raise ValueError("targets must contain at least one TPC bin.")
    return target
