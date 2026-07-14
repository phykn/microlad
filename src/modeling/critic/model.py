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
        num_phases: int,
        *,
        noise_ch: int = 128,
        base_ch: int = 64,
    ) -> None:
        super().__init__()
        if latent_ch <= 0 or noise_ch <= 0 or base_ch <= 0:
            raise ValueError("generator channels must be positive.")
        if num_phases < 2:
            raise ValueError("num_phases must be at least two.")
        if latent_size < MIN_SLICE_SIZE:
            raise ValueError(f"latent_size must be at least {MIN_SLICE_SIZE}.")

        self.latent_ch = latent_ch
        self.latent_size = latent_size
        self.num_phases = num_phases
        self.noise_ch = noise_ch
        steps = math.ceil(math.log2(latent_size / 4))
        start_ch = base_ch * 2 ** min(steps, 3)
        self.start_ch = start_ch
        self.project = nn.Linear(noise_ch + num_phases, start_ch * 4 * 4)
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

    def forward(
        self,
        noise: torch.Tensor,
        fractions: torch.Tensor,
    ) -> torch.Tensor:
        _validate_fractions(fractions, self.num_phases)
        if noise.ndim != 2 or noise.shape[1] != self.noise_ch:
            raise ValueError(
                f"noise must have shape [B, {self.noise_ch}]."
            )
        if noise.shape[0] != fractions.shape[0]:
            raise ValueError("noise and fractions batch sizes must match.")
        if noise.device != fractions.device or noise.dtype != fractions.dtype:
            raise ValueError("noise and fractions must share device and dtype.")

        inputs = torch.cat((noise, fractions), dim=1)
        hidden = self.project(inputs).reshape(-1, self.start_ch, 4, 4)
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
    """Scores phase-conditioned 2D VAE latent slices."""

    def __init__(
        self,
        latent_ch: int,
        num_phases: int,
        *,
        base_ch: int = 64,
    ) -> None:
        super().__init__()
        if latent_ch <= 0:
            raise ValueError("latent_ch must be positive.")
        if num_phases < 2:
            raise ValueError("num_phases must be at least two.")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive.")

        self.latent_ch = latent_ch
        self.num_phases = num_phases
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
        self.condition = nn.Linear(num_phases, channels[-1], bias=False)

    def forward(
        self,
        latent_slices: torch.Tensor,
        fractions: torch.Tensor,
    ) -> torch.Tensor:
        _validate_fractions(fractions, self.num_phases)
        if latent_slices.ndim != 4 or latent_slices.shape[1] != self.latent_ch:
            raise ValueError(
                "latent_slices must have shape "
                f"[B, {self.latent_ch}, H, W]."
            )
        if latent_slices.shape[0] != fractions.shape[0]:
            raise ValueError("latent slices and fractions batch sizes must match.")
        if latent_slices.device != fractions.device:
            raise ValueError("latent slices and fractions must share a device.")
        if latent_slices.dtype != fractions.dtype:
            raise ValueError("latent slices and fractions must share a dtype.")
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
        projection = (features * self.condition(fractions)).sum(dim=1, keepdim=True)
        return self.score(features) + projection * features.shape[1] ** -0.5


def _validate_fractions(fractions: torch.Tensor, num_phases: int) -> None:
    if fractions.ndim != 2 or fractions.shape[1] != num_phases:
        raise ValueError(f"fractions must have shape [B, {num_phases}].")
    if not fractions.is_floating_point():
        raise ValueError("fractions must contain finite floating-point values.")
    if fractions.device.type == "meta":
        return
    if not torch.isfinite(fractions).all():
        raise ValueError("fractions must contain finite floating-point values.")
    if torch.any(fractions < 0.0):
        raise ValueError("fractions must be non-negative.")
    if not torch.allclose(
        fractions.sum(dim=1),
        torch.ones_like(fractions[:, 0]),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("fractions must sum to one.")


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
