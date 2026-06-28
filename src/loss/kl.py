import torch


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape.")
    if mu.numel() == 0:
        raise ValueError("mu and logvar must be non-empty.")
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
