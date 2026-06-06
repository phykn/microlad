import torch
import torch.nn.functional as F


def diffusion_noise_loss(pred_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_noise, noise)
