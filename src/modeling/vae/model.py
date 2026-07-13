import torch
import torch.nn as nn

from src.modeling.normalization import norm_groups
from src.modeling.vae.spatial import downsample_steps
from src.modeling.phases.representation import (
    logits_to_probabilities,
    logits_to_relaxed_labels,
)


def get_downsample_factor(vae: nn.Module) -> int:
    factor = int(
        getattr(
            vae,
            "downsample_factor",
            int(vae.image_size) // int(vae.latent_size),
        )
    )

    if factor <= 0:
        raise ValueError("VAE downsample factor must be positive.")

    if int(vae.image_size) != int(vae.latent_size) * factor:
        raise ValueError(
            "vae.image_size must equal vae.latent_size times downsample factor."
        )

    return factor


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(norm_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups(channels), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.view(batch, channels, -1).permute(0, 2, 1)
        k = k.view(batch, channels, -1)
        v = v.view(batch, channels, -1).permute(0, 2, 1)

        attn = torch.softmax(torch.bmm(q, k) * (channels**-0.5), dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1)
        out = out.reshape(batch, channels, height, width)
        return x + self.proj_out(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.res2 = ResidualBlock(in_ch)
        self.down = ConvBlock(
            in_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.res2 = ResidualBlock(in_ch)
        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.norm = nn.GroupNorm(norm_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        x = self.act(self.norm(self.up(x)))
        return x


class PatchVAE(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        latent_size: int = 16,
        latent_ch: int = 4,
        num_phases: int = 3,
        base_ch: int = 64,
        max_ch: int = 512,
    ) -> None:
        super().__init__()

        if latent_ch <= 0:
            raise ValueError("latent_ch must be positive.")

        if base_ch <= 0:
            raise ValueError("base_ch must be positive.")

        if max_ch < base_ch:
            raise ValueError("max_ch must be greater than or equal to base_ch.")

        if num_phases < 2:
            raise ValueError("num_phases must be at least 2.")

        self.image_size = image_size
        self.latent_size = latent_size
        self.latent_ch = latent_ch
        self.num_phases = num_phases
        self.downsample_steps = downsample_steps(image_size, latent_size)
        self.downsample_factor = image_size // latent_size

        channels = [base_ch]
        for step in range(self.downsample_steps):
            channels.append(min(base_ch * 2 ** (step + 1), max_ch))
        self.channels = tuple(channels)

        self.conv_in = ConvBlock(1, channels[0])
        self.down_blocks = nn.ModuleList(
            DownBlock(channels[i], channels[i + 1])
            for i in range(self.downsample_steps)
        )
        self.enc_mid = nn.Sequential(
            ResidualBlock(channels[-1]),
            AttentionBlock(channels[-1]),
            ResidualBlock(channels[-1]),
        )
        self.to_mu = nn.Conv2d(channels[-1], latent_ch, kernel_size=1)
        self.to_logvar = nn.Conv2d(channels[-1], latent_ch, kernel_size=1)

        self.from_latent = ConvBlock(latent_ch, channels[-1])
        self.dec_mid = nn.Sequential(
            ResidualBlock(channels[-1]),
            AttentionBlock(channels[-1]),
            ResidualBlock(channels[-1]),
        )
        self.up_blocks = nn.ModuleList(
            UpBlock(channels[i], channels[i - 1])
            for i in range(len(channels) - 1, 0, -1)
        )
        self.conv_out = nn.Conv2d(channels[0], num_phases, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("input must have shape [B, 1, H, W].")

        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"input image size must be {self.image_size}x{self.image_size}."
            )

        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.enc_mid(h)
        return self.to_mu(h), self.to_logvar(h)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4 or z.shape[1] != self.latent_ch:
            raise ValueError(
                f"latent must have shape [B, {self.latent_ch}, H, W]."
            )

        if z.shape[-2:] != (self.latent_size, self.latent_size):
            raise ValueError(
                f"latent spatial size must be {self.latent_size}x{self.latent_size}."
            )

        h = self.from_latent(z)
        h = self.dec_mid(h)
        for block in self.up_blocks:
            h = block(h)
        return self.conv_out(h)

    def decode_probs(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decode_logits(z)
        return logits_to_probabilities(logits, self.num_phases)

    def decode_relaxed(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decode_logits(z)
        return logits_to_relaxed_labels(logits, self.num_phases)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_relaxed(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode_logits(z), mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape.")

    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std
