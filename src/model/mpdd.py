import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import SelfAttention, TimeEmbedding, TimeResStack


class _AnchorEncoder(nn.Module):
    def __init__(
        self,
        num_phases: int,
        base_ch: int,
        time_dim: int,
    ) -> None:
        super().__init__()
        channels = (
            max(round(base_ch * 3 / 8), 4),
            max(round(base_ch * 5 / 8), 4),
            max(base_ch, 4),
            max(round(base_ch * 3 / 2), 4),
        )
        self.input = nn.Conv2d(num_phases + 1, channels[0], 3, padding=1)
        self.block1 = TimeResStack(channels[0], channels[0], time_dim)
        self.down1 = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1)
        self.block2 = TimeResStack(channels[1], channels[1], time_dim)
        self.down2 = nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1)
        self.block3 = TimeResStack(channels[2], channels[2], time_dim)
        self.attention = SelfAttention(channels[2])
        self.down3 = nn.Conv2d(channels[2], channels[3], 4, stride=2, padding=1)
        self.block4 = TimeResStack(channels[3], channels[3], time_dim)
        self.outputs = nn.ModuleList(
            [
                nn.Conv2d(channels[0], base_ch, 1),
                nn.Conv2d(channels[1], base_ch * 2, 1),
                nn.Conv2d(channels[2], base_ch * 4, 1),
                nn.Conv2d(channels[3], base_ch * 8, 1),
            ]
        )
        for output in self.outputs:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(
        self,
        anchor: torch.Tensor,
        mask: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        support = F.max_pool2d(mask, kernel_size=9, stride=1, padding=4)
        feature1 = self.block1(
            self.input(torch.cat([anchor, mask], dim=1)),
            embedding,
        )
        feature2 = self.block2(self.down1(feature1), embedding)
        feature3 = self.attention(self.block3(self.down2(feature2), embedding))
        feature4 = self.block4(self.down3(feature3), embedding)
        features = (feature1, feature2, feature3, feature4)
        return tuple(
            output(feature)
            * F.adaptive_max_pool2d(support, output_size=feature.shape[-2:])
            for output, feature in zip(self.outputs, features, strict=True)
        )


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
        self.axis_emb = nn.Embedding(3, time_dim)
        self.anchor_encoder = _AnchorEncoder(
            num_phases,
            base_ch,
            time_dim,
        )

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
        axis_condition: torch.Tensor | None = None,
        *,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
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
        emb = emb + self._embed_axis(x, axis_condition)
        anchor_features = self._build_anchor_features(
            x,
            emb,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        e1 = self.enc1(x, emb)
        if anchor_features is not None:
            e1 = e1 + anchor_features[0]
        e2 = self.enc2(self.down1(e1), emb)
        if anchor_features is not None:
            e2 = e2 + anchor_features[1]
        e3 = self.attn3(self.enc3(self.down2(e2), emb))
        if anchor_features is not None:
            e3 = e3 + anchor_features[2]
        h = self.attn_mid(self.mid(self.down3(e3), emb))
        if anchor_features is not None:
            h = h + anchor_features[3]
        h = self.attn_dec3(self.dec3(torch.cat([self.up3(h), e3], dim=1), emb))
        h = self.dec2(torch.cat([self.up2(h), e2], dim=1), emb)
        h = self.dec1(torch.cat([self.up1(h), e1], dim=1), emb)
        return self.out(F.silu(h))

    def _build_anchor_features(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        *,
        anchor_image: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (anchor_image is None) != (anchor_mask is None):
            raise ValueError("anchor_image and anchor_mask must be provided together.")
        if anchor_image is None or anchor_mask is None:
            return None
        if anchor_image.shape != x.shape:
            raise ValueError("anchor_image must have the same shape as image batch.")
        expected_mask = (x.shape[0], 1, *x.shape[-2:])
        if anchor_mask.shape != expected_mask:
            raise ValueError("anchor_mask must have shape [B, 1, H, W].")
        if not torch.isfinite(anchor_image).all():
            raise ValueError("anchor_image must be finite.")
        if anchor_mask.dtype != torch.bool and not torch.isfinite(anchor_mask).all():
            raise ValueError("anchor_mask must be finite.")

        mask = anchor_mask.to(device=x.device, dtype=x.dtype)
        if bool(((mask < 0.0) | (mask > 1.0)).any().item()):
            raise ValueError("anchor_mask values must be between zero and one.")
        active = mask.flatten(start_dim=1).any(dim=1)
        if not bool(active.any().item()):
            return None
        selected = active.nonzero(as_tuple=False).flatten()
        mask = mask.index_select(0, selected)
        clean_condition = (
            anchor_image.to(device=x.device, dtype=x.dtype) + 1.0
        ) * 0.5
        clean_condition = clean_condition.index_select(0, selected) * mask
        features = self.anchor_encoder(
            clean_condition,
            mask,
            embedding.index_select(0, selected),
        )
        expanded = []
        for feature in features:
            empty = feature.new_zeros((x.shape[0], *feature.shape[1:]))
            expanded.append(empty.index_copy(0, selected, feature))
        return tuple(expanded)

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

    def _embed_axis(
        self,
        x: torch.Tensor,
        axis_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if axis_condition is None:
            raise ValueError("axis_condition is required.")
        if not isinstance(axis_condition, torch.Tensor):
            raise TypeError("axis_condition must be a tensor.")
        if axis_condition.shape != (x.shape[0],):
            raise ValueError("axis_condition must have shape [B].")
        if axis_condition.dtype != torch.long:
            raise TypeError("axis_condition must have dtype torch.long.")
        if bool(((axis_condition < 0) | (axis_condition >= 3)).any().item()):
            raise ValueError("axis_condition values must be in the range 0 to 2.")
        return self.axis_emb(axis_condition.to(device=x.device))
