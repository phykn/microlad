import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MIN_SLICE_SIZE = 16


class LatentGenerator(nn.Module):
    def __init__(
        self,
        latent_ch: int,
        latent_size: int,
        *,
        noise_ch: int = 128,
        base_ch: int = 64,
    ) -> None:
        super().__init__()
        if latent_ch <= 0 or noise_ch <= 0 or base_ch <= 0:
            raise ValueError("generator channels must be positive.")
        if latent_size < MIN_SLICE_SIZE:
            raise ValueError(f"latent_size must be at least {MIN_SLICE_SIZE}.")

        self.latent_ch = latent_ch
        self.latent_size = latent_size
        self.noise_ch = noise_ch
        steps = math.ceil(math.log2(latent_size / 4))
        start_ch = base_ch * 2 ** min(steps, 3)
        self.start_ch = start_ch
        self.project = nn.Linear(noise_ch, start_ch * 4 * 4)
        blocks = []
        channels = start_ch
        for _ in range(steps):
            next_ch = max(base_ch, channels // 2)
            blocks.extend(
                (
                    nn.ConvTranspose2d(
                        channels,
                        next_ch,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_groups(next_ch), next_ch),
                    nn.ReLU(inplace=True),
                )
            )
            channels = next_ch
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Conv2d(channels, latent_ch, kernel_size=3, padding=1)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.noise_ch:
            raise ValueError(
                f"noise must have shape [B, {self.noise_ch}]."
            )

        hidden = self.project(noise).reshape(-1, self.start_ch, 4, 4)
        hidden = self.blocks(hidden)
        if hidden.shape[-2:] != (self.latent_size, self.latent_size):
            hidden = F.interpolate(
                hidden,
                size=(self.latent_size, self.latent_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.output(hidden)


class LatentCritic(nn.Module):
    """Scores 2D VAE latent slices."""

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


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
