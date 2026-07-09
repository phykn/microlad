import torch
import torch.nn as nn

from src.loss.kl import kl_divergence
from src.loss.phase import logits_to_phase_values, phase_cross_entropy
from src.loss.ssim import ssim_loss


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    ssim_weight: float = 0.1,
    num_phases: int = 3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if recon.ndim != 4:
        raise ValueError("recon must have shape [B, num_phases, H, W].")

    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("target must have shape [B, 1, H, W].")

    if recon.shape[1] != num_phases:
        raise ValueError("recon channel count must match num_phases.")

    if recon.shape[0] != target.shape[0] or recon.shape[-2:] != target.shape[-2:]:
        raise ValueError("recon and target spatial shape must match.")

    if recon.numel() == 0 or target.numel() == 0:
        raise ValueError("recon and target must not be empty.")

    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape.")

    if mu.ndim == 0 or mu.shape[0] != recon.shape[0]:
        raise ValueError("mu and logvar batch size must match recon batch size.")

    if beta < 0:
        raise ValueError("beta must be non-negative.")

    if ssim_weight < 0:
        raise ValueError("ssim_weight must be non-negative.")

    reconstruction = phase_cross_entropy(recon, target, num_phases)
    kl = kl_divergence(mu, logvar)
    recon_values = logits_to_phase_values(recon, num_phases)
    structural = (
        ssim_loss(recon_values, target, data_range=float(num_phases - 1))
        if ssim_weight > 0
        else torch.zeros((), device=recon.device, dtype=recon.dtype)
    )
    total = reconstruction + ssim_weight * structural + beta * kl
    parts = {
        "reconstruction": reconstruction.detach(),
        "ssim": structural.detach(),
        "kl": kl.detach(),
    }
    return total, parts


class VAELoss(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        ssim_weight: float = 0.1,
        num_phases: int = 3,
    ) -> None:
        super().__init__()

        if beta < 0:
            raise ValueError("beta must be non-negative.")

        if ssim_weight < 0:
            raise ValueError("ssim_weight must be non-negative.")

        self.beta = beta
        self.ssim_weight = ssim_weight
        self.num_phases = num_phases

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
            num_phases=self.num_phases,
        )
