import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.diffusion.process import DDPMProcess


def diffusion_loss(
    model: nn.Module,
    ddpm: DDPMProcess,
    clean_latent: torch.Tensor,
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if clean_latent.ndim != 4:
        raise ValueError("clean_latent must have shape [B, C, H, W].")

    if any(size <= 0 for size in clean_latent.shape):
        raise ValueError("clean_latent dimensions must be positive.")

    if t is None:
        t = ddpm.sample_timesteps(clean_latent.shape[0], device=clean_latent.device)
    elif t.device != clean_latent.device:
        raise ValueError("t must be on the same device as clean_latent.")

    if noise is None:
        noise = torch.randn_like(clean_latent)

    if noise.shape != clean_latent.shape:
        raise ValueError("noise must have the same shape as clean_latent.")

    noisy_latent = ddpm.add_noise(clean_latent, t, noise=noise)
    pred_noise = model(noisy_latent, t)

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
        clean_latent: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return diffusion_loss(
            model,
            self.ddpm,
            clean_latent,
            t=t,
            noise=noise,
        )
