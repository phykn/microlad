import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


def _ssim_loss(
    recon: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
) -> torch.Tensor:
    return 1.0 - ssim(recon, target, data_range=data_range, size_average=True)


def _loss_result(
    loss: torch.Tensor, parts: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return loss, {name: value.detach() for name, value in parts.items()}


def compute_vae_loss_parts(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6,
    ssim_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction = F.mse_loss(recon, target)
    ssim_loss = _ssim_loss(recon, target)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.numel()
    total = reconstruction + ssim_weight * ssim_loss + kl_weight * kl
    return total, {"reconstruction": reconstruction, "ssim": ssim_loss, "kl": kl}


class VAELoss(nn.Module):
    def __init__(self, kl_weight: float = 1e-6, ssim_weight: float = 0.1) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.ssim_weight = ssim_weight

    def forward(
        self,
        model: torch.nn.Module,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = next(model.parameters()).device
        image = batch.to(device)
        recon, mu, logvar = model(image * 2 - 1)
        total, parts = compute_vae_loss_parts(
            recon,
            image,
            mu,
            logvar,
            kl_weight=self.kl_weight,
            ssim_weight=self.ssim_weight,
        )
        return _loss_result(total, parts)


class UNetDiffusionLoss(nn.Module):
    def __init__(self, vae: nn.Module, ddpm) -> None:
        super().__init__()
        self.vae = vae
        self.ddpm = ddpm

    def forward(
        self, model: nn.Module, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = next(model.parameters()).device
        image = batch.to(device)

        with torch.no_grad():
            z, _ = self.vae.encode(image * 2.0 - 1.0)

        t = torch.randint(
            0, self.ddpm.num_timesteps, (z.shape[0],), device=device, dtype=torch.long
        )
        noise = torch.randn_like(z)
        z_t = self.ddpm.q_sample(z, t, noise)
        pred_noise = model(z_t, t)
        loss = F.mse_loss(pred_noise, noise)
        return _loss_result(loss, {})
