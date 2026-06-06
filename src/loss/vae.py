import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = reconstruction + kl_weight * kl
    return total, {"reconstruction": reconstruction.detach(), "kl": kl.detach()}


class VAELoss(nn.Module):
    def __init__(self, kl_weight: float = 1e-6) -> None:
        super().__init__()
        self.kl_weight = kl_weight

    def forward(
        self,
        model: torch.nn.Module,
        batch: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(model.parameters()).device
        target = batch.to(device)
        recon, mu, logvar = model(target * 2 - 1)
        total, parts = vae_loss(recon, target, mu, logvar, kl_weight=self.kl_weight)
        return {
            "loss": total.detach(),
            "reconstruction": parts["reconstruction"],
            "kl": parts["kl"],
        }, total
