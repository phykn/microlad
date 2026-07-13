import torch
import torch.nn as nn
import torch.nn.functional as F


MIN_SLICE_SIZE = 16


class LatentCritic(nn.Module):
    """Scores 2D VAE latent slices."""

    def __init__(self, latent_ch: int, *, base_ch: int = 64) -> None:
        super().__init__()
        if latent_ch <= 0:
            raise ValueError("latent_ch must be positive.")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive.")

        self.latent_ch = latent_ch
        channels = (latent_ch, base_ch, base_ch * 2, 1)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    source,
                    target,
                    kernel_size=4,
                    stride=2,
                    padding=1 if index < 2 else 0,
                    bias=False,
                )
                for index, (source, target) in enumerate(zip(channels, channels[1:]))
            ]
        )

    def forward(self, latent_slices: torch.Tensor) -> torch.Tensor:
        if latent_slices.ndim != 4 or latent_slices.shape[1] != self.latent_ch:
            raise ValueError(
                "latent_slices must have shape "
                f"[B, {self.latent_ch}, H, W]."
            )
        if min(latent_slices.shape[-2:]) < MIN_SLICE_SIZE:
            raise ValueError(
                f"latent slice size must be at least {MIN_SLICE_SIZE}."
            )

        scores = latent_slices
        for layer in self.layers[:-1]:
            scores = F.leaky_relu(layer(scores), negative_slope=0.2)
        return self.layers[-1](scores)
