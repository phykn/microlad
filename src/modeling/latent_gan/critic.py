import torch
import torch.nn as nn


MIN_SLICE_SIZE = 16


class LatentCritic(nn.Module):
    """Scores 2D VAE latent feature maps."""

    def __init__(
        self,
        latent_ch: int,
        *,
        base_ch: int = 64,
    ) -> None:
        super().__init__()
        if latent_ch <= 0:
            raise ValueError("latent_ch must be positive.")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive.")

        self.latent_ch = latent_ch
        channels = (latent_ch, base_ch, base_ch * 2, base_ch * 4)
        layers = []
        for source, target in zip(channels, channels[1:]):
            layers.extend(
                (
                    nn.Conv2d(
                        source,
                        target,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.features = nn.Sequential(*layers)
        self.score = nn.Linear(channels[-1], 1)

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

        centered = latent_slices - latent_slices.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        scale = centered.square().mean(
            dim=(1, 2, 3),
            keepdim=True,
        )
        normalized = centered * torch.rsqrt(scale + 1e-6)
        features = self.features(normalized).mean(dim=(-2, -1))
        return self.score(features)
