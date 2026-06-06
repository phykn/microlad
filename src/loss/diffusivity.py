import torch
import torch.nn.functional as F


def compute_diffusivity_loss(
    masks: torch.Tensor,
    fem_solver: torch.nn.Module,
    rd_targets: dict[int, float],
    phases: list[int],
    device: torch.device,
) -> torch.Tensor:
    if masks.ndim != 4:
        raise ValueError("masks must have shape [B, P, H, W].")
    if masks.shape[0] != 1:
        raise ValueError("compute_diffusivity_loss currently expects batch size 1.")

    deff = []
    for phase_index in range(len(phases)):
        deff.append(fem_solver(masks[0, phase_index]))
    pred = torch.stack(deff)
    target = torch.tensor([rd_targets[phase] for phase in phases], device=device, dtype=pred.dtype)
    return F.mse_loss(pred, target)
