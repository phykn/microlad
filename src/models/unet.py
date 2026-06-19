import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(channels: int) -> int:
    for groups in range(min(16, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.act1 = nn.SiLU()
        self.lin2 = nn.Linear(dim * 4, dim * 4)
        self.act2 = nn.SiLU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = max(self.dim // 2, 1)
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        elif emb.shape[-1] > self.dim:
            emb = emb[:, : self.dim]
        h = self.act1(self.lin1(emb))
        return self.act2(self.lin2(h))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = nn.GroupNorm(_norm_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim * 4, out_ch)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(self.norm1(x))) + self.time_mlp(te)[:, :, None, None]
        h = F.silu(self.conv2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_norm_groups(ch), ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = nn.Conv1d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = self.norm(x).view(b, c, -1)
        q, k, v = self.qkv(flat).chunk(3, dim=1)
        q, k, v = [item.permute(0, 2, 1) for item in (q, k, v)]
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(c), dim=-1)
        out = (attn @ v).permute(0, 2, 1)
        return x + self.proj(out).view(b, c, h, w)


class TimeUNet(nn.Module):
    """Time-conditioned UNet for latent-space noise prediction."""

    def __init__(self, latent_ch: int, base_ch: int = 128, time_dim: int = 64) -> None:
        if latent_ch <= 0:
            raise ValueError("latent_ch must be positive.")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive.")
        if time_dim <= 0:
            raise ValueError("time_dim must be positive.")

        super().__init__()
        self.latent_ch = latent_ch
        self.time_emb = _TimeEmbedding(time_dim)
        self.enc1 = ResidualBlock(latent_ch, base_ch, time_dim)
        self.attn16 = SelfAttention(base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_ch, base_ch * 2, time_dim)
        self.attn8 = SelfAttention(base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(base_ch * 2, base_ch * 4, time_dim)
        self.attn4 = SelfAttention(base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = ResidualBlock(base_ch * 4, base_ch * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = ResidualBlock(base_ch * 2, base_ch, time_dim)
        self.out = nn.Conv2d(base_ch, latent_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.latent_ch:
            raise ValueError(
                f"latent batch must have shape [B, {self.latent_ch}, H, W]."
            )
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("timesteps must have shape [B].")
        if x.shape[-2] % 4 != 0 or x.shape[-1] % 4 != 0:
            raise ValueError("latent height and width must be divisible by 4.")

        te = self.time_emb(t)
        e1 = self.attn16(self.enc1(x, te))
        e2 = self.attn8(self.enc2(self.pool1(e1), te))
        b = self.attn4(self.bottleneck(self.pool2(e2), te))
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1), te)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1), te)
        return self.out(d1)
