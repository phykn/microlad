import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.kl import kl_divergence
from src.loss.phase import phase_loss
from src.loss.ssim import ssim_loss


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    ssim_weight: float = 0.1,
    phase_weight: float = 0.1,
    num_phases: int = 3,
    phase_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if recon.shape != target.shape:
        raise ValueError("recon and target must have the same shape.")
    if recon.numel() == 0:
        raise ValueError("recon and target must not be empty.")
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape.")
    if mu.ndim == 0 or mu.shape[0] != recon.shape[0]:
        raise ValueError("mu and logvar batch size must match recon batch size.")
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    if ssim_weight < 0:
        raise ValueError("ssim_weight must be non-negative.")
    if phase_weight < 0:
        raise ValueError("phase_weight must be non-negative.")

    reconstruction = F.mse_loss(recon, target)
    kl = kl_divergence(mu, logvar)
    structural = (
        ssim_loss(recon, target)
        if ssim_weight > 0
        else torch.zeros((), device=recon.device, dtype=recon.dtype)
    )
    phase = (
        phase_loss(recon, target, num_phases, phase_temperature)
        if phase_weight > 0
        else torch.zeros((), device=recon.device, dtype=recon.dtype)
    )
    total = reconstruction + ssim_weight * structural + phase_weight * phase + beta * kl
    parts = {
        "reconstruction": reconstruction.detach(),
        "ssim": structural.detach(),
        "phase": phase.detach(),
        "kl": kl.detach(),
    }
    return total, parts


class VAELoss(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        ssim_weight: float = 0.1,
        phase_weight: float = 0.1,
        num_phases: int = 3,
        phase_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if ssim_weight < 0:
            raise ValueError("ssim_weight must be non-negative.")
        if phase_weight < 0:
            raise ValueError("phase_weight must be non-negative.")
        self.beta = beta
        self.ssim_weight = ssim_weight
        self.phase_weight = phase_weight
        self.num_phases = num_phases
        self.phase_temperature = phase_temperature

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return vae_loss(
            recon,
            target,
            mu,
            logvar,
            beta=self.beta,
            ssim_weight=self.ssim_weight,
            phase_weight=self.phase_weight,
            num_phases=self.num_phases,
            phase_temperature=self.phase_temperature,
        )
