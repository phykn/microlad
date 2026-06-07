import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import compute_vae_loss_parts


def diffusion_noise_loss(pred_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_noise, noise)


class VAELoss(nn.Module):
    def __init__(self, kl_weight: float = 1e-6, ssim_weight: float = 0.1) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.ssim_weight = ssim_weight

    def forward(
        self,
        model: torch.nn.Module,
        batch: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(model.parameters()).device
        target = batch.to(device)
        recon, mu, logvar = model(target * 2 - 1)
        total, parts = compute_vae_loss_parts(
            recon,
            target,
            mu,
            logvar,
            kl_weight=self.kl_weight,
            ssim_weight=self.ssim_weight,
        )
        return {
            "loss": total.detach(),
            "reconstruction": parts["reconstruction"],
            "ssim": parts["ssim"],
            "kl": parts["kl"],
        }, total


class UNetDiffusionLoss(nn.Module):
    """Trainer criterion for 2D latent diffusion noise prediction."""

    def __init__(self, vae: nn.Module, ddpm) -> None:
        super().__init__()
        self.vae = vae
        self.ddpm = ddpm

    def forward(self, model: nn.Module, batch: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(model.parameters()).device
        target = batch.to(device)

        with torch.no_grad():
            z, _ = self.vae.encode(target * 2.0 - 1.0)

        t = torch.randint(0, self.ddpm.num_timesteps, (z.shape[0],), device=device, dtype=torch.long)
        noise = torch.randn_like(z)
        z_t = self.ddpm.q_sample(z, t, noise)
        pred_noise = model(z_t, t)
        loss = diffusion_noise_loss(pred_noise, noise)
        return {"loss": loss.detach(), "diffusion": loss.detach()}, loss
