import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _msssim
except ImportError:
    _msssim = None


def _ssim_loss(recon: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    if _msssim is not None and recon.shape[-1] >= 11 and recon.shape[-2] >= 11:
        return 1.0 - _msssim(recon, target, data_range=data_range, size_average=True)

    dims = (-2, -1)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    recon_mean = recon.mean(dim=dims, keepdim=True)
    target_mean = target.mean(dim=dims, keepdim=True)
    recon_var = (recon - recon_mean).pow(2).mean(dim=dims, keepdim=True)
    target_var = (target - target_mean).pow(2).mean(dim=dims, keepdim=True)
    covariance = ((recon - recon_mean) * (target - target_mean)).mean(dim=dims, keepdim=True)

    numerator = (2 * recon_mean * target_mean + c1) * (2 * covariance + c2)
    denominator = (recon_mean.pow(2) + target_mean.pow(2) + c1) * (recon_var + target_var + c2)
    ssim = (numerator / denominator.clamp_min(1e-12)).mean()
    return 1.0 - ssim.clamp(-1.0, 1.0)


def compute_vae_loss_parts(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6,
    ssim_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction = F.mse_loss(recon, target)
    ssim = _ssim_loss(recon, target)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.numel()
    total = reconstruction + ssim_weight * ssim + kl_weight * kl
    return total, {"reconstruction": reconstruction.detach(), "ssim": ssim.detach(), "kl": kl.detach()}
