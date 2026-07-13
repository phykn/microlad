import torch
import torch.nn as nn
import torch.nn.functional as F


NOISE_CHANNELS = 32
SCALE_FACTOR = 4
MIN_SLICE_SIZE = 16


def output_size(noise_size: int) -> int:
    if (
        not isinstance(noise_size, int)
        or isinstance(noise_size, bool)
        or noise_size <= 0
    ):
        raise ValueError("noise_size must be a positive integer.")
    return noise_size * SCALE_FACTOR


class SliceGANGenerator(nn.Module):
    """Generate an unbounded 3D VAE latent field from spatial noise."""

    def __init__(
        self,
        latent_ch: int,
        *,
        noise_ch: int = NOISE_CHANNELS,
        base_ch: int = 256,
    ) -> None:
        super().__init__()
        if latent_ch <= 0:
            raise ValueError("latent_ch must be positive.")
        if noise_ch <= 0:
            raise ValueError("noise_ch must be positive.")
        if base_ch < 2:
            raise ValueError("base_ch must be at least 2.")

        self.latent_ch = latent_ch
        self.noise_ch = noise_ch
        self.scale_factor = SCALE_FACTOR
        channels = (noise_ch, base_ch, base_ch // 2)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose3d(
                        source,
                        target,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(target),
                    nn.ReLU(inplace=True),
                )
                for source, target in zip(channels, channels[1:])
            ]
        )
        self.to_latent = nn.Conv3d(
            channels[-1],
            latent_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 5 or noise.shape[1] != self.noise_ch:
            raise ValueError(
                f"noise must have shape [B, {self.noise_ch}, D, H, W]."
            )
        if any(size <= 0 for size in noise.shape):
            raise ValueError("noise dimensions must be positive.")

        latent = noise
        for block in self.blocks:
            latent = block(latent)
        return self.to_latent(latent)


class SliceGANCritic(nn.Module):
    """Score 2D slices from real or generated VAE latent fields."""

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
            scores = F.relu(layer(scores))
        return self.layers[-1](scores)
