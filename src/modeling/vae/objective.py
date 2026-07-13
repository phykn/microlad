import math

import torch
import torch.nn as nn

from src.modeling.phases.representation import phase_cross_entropy


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Return KL divergence averaged over every latent element."""
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape.")

    if mu.numel() == 0:
        raise ValueError("mu and logvar must be non-empty.")

    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    num_phases: int = 3,
    phase_balance: float = 0.0,
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

    reconstruction = phase_cross_entropy(
        recon,
        target,
        num_phases,
        phase_balance=phase_balance,
    )
    kl = kl_divergence(mu, logvar)
    total = reconstruction + beta * kl
    parts = {
        "reconstruction": reconstruction.detach(),
        "kl": kl.detach(),
    }
    return total, parts


class VAELoss(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        num_phases: int = 3,
        phase_balance: float = 0.0,
    ) -> None:
        super().__init__()

        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if (
            not isinstance(phase_balance, (int, float))
            or isinstance(phase_balance, bool)
            or not math.isfinite(phase_balance)
            or phase_balance < 0.0
            or phase_balance > 1.0
        ):
            raise ValueError("phase_balance must be between zero and one.")

        self.beta = beta
        self.num_phases = num_phases
        self.phase_balance = phase_balance

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
            num_phases=self.num_phases,
            phase_balance=self.phase_balance,
        )
