import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DDPM


def diffusion_noise_loss(pred_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_noise, noise)


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


class SliceConditionedDiffusionLoss(nn.Module):
    def __init__(
        self,
        vae: torch.nn.Module,
        ddpm: DDPM,
        condition_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be between 0 and 1.")
        self.vae = vae
        self.ddpm = ddpm
        self.condition_dropout = condition_dropout

    def forward(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(model.parameters()).device
        target = batch["target"].to(device)
        condition = batch["condition"].to(device)
        axis = batch["axis"].to(device).long()
        slice_index = batch["slice_index"].to(device).long()

        with torch.no_grad():
            target_z, _ = self.vae.encode(target * 2 - 1)
            condition_z, _ = self.vae.encode(condition * 2 - 1)

        dropout_mask = torch.rand(target_z.shape[0], device=device) < self.condition_dropout
        if dropout_mask.any():
            condition_z = condition_z.clone()
            axis = axis.clone()
            slice_index = slice_index.clone()
            condition_z[dropout_mask] = 0.0
            axis[dropout_mask] = model.null_axis
            slice_index[dropout_mask] = model.null_slice

        t = torch.randint(0, self.ddpm.num_timesteps, (target_z.shape[0],), device=device)
        noise = torch.randn_like(target_z)
        z_t = self.ddpm.q_sample(target_z, t, noise)
        pred_noise = model(z_t, t, condition_z, axis, slice_index)
        loss = diffusion_noise_loss(pred_noise, noise)
        dropout_rate = dropout_mask.float().mean().detach()
        return {
            "loss": loss.detach(),
            "diffusion": loss.detach(),
            "condition_dropout": dropout_rate,
        }, loss


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
