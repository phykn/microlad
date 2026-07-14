import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.normalization import norm_groups


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = max(self.dim // 2, 1)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb[:, : self.dim])


class TimeResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.norm1 = nn.GroupNorm(norm_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TimeResStack(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.block1 = TimeResBlock(in_ch, out_ch, time_dim)
        self.block2 = TimeResBlock(out_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, time_emb)
        return self.block2(x, time_emb)


class SelfAttention(nn.Module):
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

        attn = torch.softmax(q @ k / math.sqrt(channels), dim=-1)
        out = (attn @ v).permute(0, 2, 1)
        out = out.reshape(batch, channels, height, width)
        return x + self.proj_out(out)


class TimeUNet(nn.Module):
    def __init__(
        self,
        latent_ch: int,
        base_ch: int = 128,
        time_dim: int = 64,
        num_phases: int = 3,
    ) -> None:
        super().__init__()

        if latent_ch <= 0:
            raise ValueError("latent_ch must be positive.")

        if base_ch <= 0:
            raise ValueError("base_ch must be positive.")

        if time_dim <= 0:
            raise ValueError("time_dim must be positive.")
        if num_phases < 2:
            raise ValueError("num_phases must be at least 2.")

        self.latent_ch = latent_ch
        self.base_ch = base_ch
        self.time_dim = time_dim
        self.num_phases = num_phases

        self.time_emb = TimeEmbedding(time_dim)
        self.fraction_emb = nn.Sequential(
            nn.Linear(num_phases, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.null_fraction_emb = nn.Parameter(torch.zeros(time_dim))
        self.enc1 = TimeResStack(latent_ch, base_ch, time_dim)
        self.attn1 = SelfAttention(base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = TimeResStack(base_ch * 2, base_ch * 2, time_dim)
        self.attn2 = SelfAttention(base_ch * 2)
        self.down2 = nn.Conv2d(
            base_ch * 2,
            base_ch * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.mid = TimeResStack(base_ch * 4, base_ch * 4, time_dim)
        self.attn_mid = SelfAttention(base_ch * 4)
        self.up2 = nn.ConvTranspose2d(
            base_ch * 4,
            base_ch * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec2 = TimeResStack(base_ch * 4, base_ch * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(
            base_ch * 2,
            base_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec1 = TimeResStack(base_ch * 2, base_ch, time_dim)
        self.out = nn.Conv2d(base_ch, latent_ch, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        phase_fractions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.latent_ch:
            raise ValueError(
                f"latent batch must have shape [B, {self.latent_ch}, H, W]."
            )

        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("timesteps must have shape [B].")

        if x.shape[0] <= 0 or x.shape[-2] <= 0 or x.shape[-1] <= 0:
            raise ValueError("latent batch dimensions must be positive.")

        if x.shape[-2] % 4 != 0 or x.shape[-1] % 4 != 0:
            raise ValueError("latent height and width must be divisible by 4.")

        if phase_fractions is None:
            fraction_emb = self.null_fraction_emb.expand(x.shape[0], -1)
        else:
            if phase_fractions.shape != (x.shape[0], self.num_phases):
                raise ValueError("phase_fractions must have shape [B, num_phases].")
            phase_fractions = phase_fractions.to(device=x.device, dtype=x.dtype)
            if not torch.isfinite(phase_fractions).all():
                raise ValueError("phase_fractions must be finite.")
            null = phase_fractions.sum(dim=1) == 0
            fraction_emb = self.fraction_emb(phase_fractions)
            fraction_emb = torch.where(
                null[:, None],
                self.null_fraction_emb.expand_as(fraction_emb),
                fraction_emb,
            )

        time_emb = self.time_emb(t) + fraction_emb
        e1 = self.attn1(self.enc1(x, time_emb))
        e2 = self.attn2(self.enc2(self.down1(e1), time_emb))
        h = self.attn_mid(self.mid(self.down2(e2), time_emb))
        h = self.dec2(torch.cat([self.up2(h), e2], dim=1), time_emb)
        h = self.dec1(torch.cat([self.up1(h), e1], dim=1), time_emb)
        return self.out(F.silu(h))
