import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddpm import DDPMProcess


def compute_loss(
    model: nn.Module,
    ddpm: DDPMProcess,
    clean: torch.Tensor,
    fractions: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if clean.ndim != 4:
        raise ValueError("clean_image must have shape [B, C, H, W].")

    if any(size <= 0 for size in clean.shape):
        raise ValueError("clean_image dimensions must be positive.")

    if t is None:
        t = ddpm.sample_steps(clean.shape[0], device=clean.device)
    elif t.device != clean.device:
        raise ValueError("t must be on the same device as clean_image.")

    if noise is None:
        noise = torch.randn_like(clean)

    if noise.shape != clean.shape:
        raise ValueError("noise must have the same shape as clean_image.")

    noisy = ddpm.add_noise(clean, t, noise=noise)
    pred_noise = model(noisy, t) if fractions is None else model(noisy, t, fractions)

    if pred_noise.shape != noise.shape:
        raise ValueError("model output must have the same shape as noise.")

    loss = F.mse_loss(pred_noise, noise)
    return loss, {"noise": loss.detach()}


class DiffusionLoss(nn.Module):
    def __init__(self, ddpm: DDPMProcess) -> None:
        super().__init__()
        self.ddpm = ddpm

    def forward(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        fractions: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return compute_loss(
            model,
            self.ddpm,
            clean,
            fractions=fractions,
            t=t,
            noise=noise,
        )
