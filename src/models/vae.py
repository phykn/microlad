import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act2(x + h)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(16, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=1)
        q = q.view(b, c, -1).permute(0, 2, 1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1).permute(0, 2, 1)
        scale = torch.sqrt(torch.tensor(c, dtype=torch.float32, device=x.device))
        attn = torch.softmax(torch.bmm(q, k) / scale, dim=-1)
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return x + self.proj_out(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.res2 = ResidualBlock(in_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.gn = nn.GroupNorm(16, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        return self.act(self.gn(self.down(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.res2 = ResidualBlock(in_ch)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.gn = nn.GroupNorm(16, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        return self.act(self.gn(self.up(x)))


class CustomVAE(nn.Module):
    """VAE for 64x64 grayscale microstructure patches."""

    def __init__(self, latent_ch: int = 4) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.down1 = DownBlock(128, 128)
        self.down2 = DownBlock(128, 256)
        self.res1 = ResidualBlock(256)
        self.attn1 = AttentionBlock(256)
        self.res2 = ResidualBlock(256)
        self.to_mu = nn.Conv2d(256, latent_ch, kernel_size=1)
        self.to_logvar = nn.Conv2d(256, latent_ch, kernel_size=1)

        self.conv_z = nn.Conv2d(latent_ch, 256, kernel_size=3, padding=1)
        self.res3 = ResidualBlock(256)
        self.attn2 = AttentionBlock(256)
        self.res4 = ResidualBlock(256)
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.act_out = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.res1(h)
        h = self.attn1(h)
        h = self.res2(h)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_z(z)
        h = self.res3(h)
        h = self.attn2(h)
        h = self.res4(h)
        h = self.up1(h)
        h = self.up2(h)
        return self.act_out(self.conv_out(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std
