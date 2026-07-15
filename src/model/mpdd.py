import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import SelfAttention, TimeEmbedding, TimeResStack


class MPDDUNet(nn.Module):
    """Image-space noise predictor used by multi-plane denoising diffusion."""

    def __init__(
        self,
        num_phases: int,
        image_size: int = 64,
        base_ch: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_phases < 2:
            raise ValueError("num_phases must be at least 2.")
        if image_size <= 0 or image_size % 8 != 0:
            raise ValueError("image_size must be positive and divisible by 8.")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive.")
        if time_dim <= 0:
            raise ValueError("time_dim must be positive.")

        self.num_phases = int(num_phases)
        self.image_size = int(image_size)
        self.base_ch = int(base_ch)
        self.time_dim = int(time_dim)

        self.time_emb = TimeEmbedding(time_dim)
        self.fraction_emb = nn.Sequential(
            nn.Linear(num_phases, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.null_fraction_emb = nn.Parameter(torch.zeros(time_dim))

        self.enc1 = TimeResStack(num_phases, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.enc2 = TimeResStack(base_ch * 2, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.enc3 = TimeResStack(base_ch * 4, base_ch * 4, time_dim)
        self.attn3 = SelfAttention(base_ch * 4)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1)

        self.mid = TimeResStack(base_ch * 8, base_ch * 8, time_dim)
        self.attn_mid = SelfAttention(base_ch * 8)

        self.up3 = nn.ConvTranspose2d(
            base_ch * 8,
            base_ch * 4,
            4,
            stride=2,
            padding=1,
        )
        self.dec3 = TimeResStack(base_ch * 8, base_ch * 4, time_dim)
        self.attn_dec3 = SelfAttention(base_ch * 4)
        self.up2 = nn.ConvTranspose2d(
            base_ch * 4,
            base_ch * 2,
            4,
            stride=2,
            padding=1,
        )
        self.dec2 = TimeResStack(base_ch * 4, base_ch * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(
            base_ch * 2,
            base_ch,
            4,
            stride=2,
            padding=1,
        )
        self.dec1 = TimeResStack(base_ch * 2, base_ch, time_dim)
        self.out = nn.Conv2d(base_ch, num_phases, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        fractions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.image_size, self.image_size)
        if x.ndim != 4 or x.shape[1] != self.num_phases:
            raise ValueError(
                f"image batch must have shape [B, {self.num_phases}, H, W]."
            )
        if tuple(x.shape[-2:]) != expected:
            raise ValueError(f"image height and width must both be {self.image_size}.")
        if x.shape[0] <= 0:
            raise ValueError("image batch must be non-empty.")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("timesteps must have shape [B].")

        emb = self.time_emb(t) + self._embed_fractions(x, fractions)
        e1 = self.enc1(x, emb)
        e2 = self.enc2(self.down1(e1), emb)
        e3 = self.attn3(self.enc3(self.down2(e2), emb))
        h = self.attn_mid(self.mid(self.down3(e3), emb))
        h = self.attn_dec3(self.dec3(torch.cat([self.up3(h), e3], dim=1), emb))
        h = self.dec2(torch.cat([self.up2(h), e2], dim=1), emb)
        h = self.dec1(torch.cat([self.up1(h), e1], dim=1), emb)
        return self.out(F.silu(h))

    def _embed_fractions(
        self,
        x: torch.Tensor,
        fractions: torch.Tensor | None,
    ) -> torch.Tensor:
        if fractions is None:
            return self.null_fraction_emb.expand(x.shape[0], -1)
        if fractions.shape != (x.shape[0], self.num_phases):
            raise ValueError("phase_fractions must have shape [B, num_phases].")
        fractions = fractions.to(device=x.device, dtype=x.dtype)
        if not torch.isfinite(fractions).all() or torch.any(fractions < 0.0):
            raise ValueError("phase_fractions must be finite and non-negative.")
        totals = fractions.sum(dim=1)
        null = totals == 0
        if bool((~null & ((totals - 1.0).abs() > 1e-4)).any().item()):
            raise ValueError("non-null phase_fractions must sum to one.")
        emb = self.fraction_emb(fractions)
        return torch.where(
            null[:, None],
            self.null_fraction_emb.expand_as(emb),
            emb,
        )
