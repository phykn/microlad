import torch
from pytorch_msssim import ssim


def ssim_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
) -> torch.Tensor:
    if recon.shape != target.shape:
        raise ValueError("recon and target must have the same shape.")
    if recon.ndim != 4:
        raise ValueError("recon and target must have shape [B, C, H, W].")
    if data_range <= 0:
        raise ValueError("data_range must be positive.")

    recon = recon.clamp(-1.0, 1.0)
    target = target.clamp(-1.0, 1.0)
    return 1.0 - ssim(recon, target, data_range=data_range, size_average=True)
