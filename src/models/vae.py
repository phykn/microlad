import torch
import torch.nn as nn

from src.models.norm import norm_groups
from src.models.shape import downsample_steps


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


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = ConvBlock(
            in_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.norm = nn.GroupNorm(norm_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.up(x)))
        return self.res(x)


class PatchVAE(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        latent_size: int = 16,
        latent_ch: int = 4,
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

        self.image_size = image_size
        self.latent_size = latent_size
        self.latent_ch = latent_ch
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
        self.enc_mid = ResidualBlock(channels[-1])
        self.to_mu = nn.Conv2d(channels[-1], latent_ch, kernel_size=1)
        self.to_logvar = nn.Conv2d(channels[-1], latent_ch, kernel_size=1)

        self.from_latent = ConvBlock(latent_ch, channels[-1])
        self.dec_mid = ResidualBlock(channels[-1])
        self.up_blocks = nn.ModuleList(
            UpBlock(channels[i], channels[i - 1])
            for i in range(len(channels) - 1, 0, -1)
        )
        self.conv_out = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std
