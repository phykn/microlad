from collections.abc import Sequence

import torch

from src.models import DDPM


class DiffusionSampler:
    def __init__(
        self,
        model: torch.nn.Module,
        ddpm: DDPM,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.ddpm = ddpm

    @torch.no_grad()
    def sample(
        self,
        shape: Sequence[int],
        *,
        anchor_latent: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = self._validate_shape(shape)
        anchor_latent, anchor_mask = self._prepare_anchor(
            shape,
            anchor_latent=anchor_latent,
            anchor_mask=anchor_mask,
        )

        self.model.eval()
        x = torch.randn(shape, device=self.device)
        batch_size = shape[0]
        for step in range(self.ddpm.num_timesteps - 1, -1, -1):
            t = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            x = self.ddpm.p_sample(self.model, x, t)
            if anchor_latent is not None and anchor_mask is not None:
                x = self._blend_anchor(x, anchor_latent, anchor_mask, step)
        return x

    @torch.no_grad()
    def sample_lmpdd(
        self,
        shape: Sequence[int],
        *,
        anchor_latent: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = self._validate_shape(shape)
        self._validate_lmpdd_shape(shape)
        anchor_latent, anchor_mask = self._prepare_anchor(
            shape,
            anchor_latent=anchor_latent,
            anchor_mask=anchor_mask,
        )

        self.model.eval()
        x = torch.randn(shape, device=self.device)
        batch_size = shape[0]
        for step in range(self.ddpm.num_timesteps - 1, -1, -1):
            t = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            x = self.ddpm.p_sample(self.model, x, t)
            if anchor_latent is not None and anchor_mask is not None:
                x = self._blend_anchor(x, anchor_latent, anchor_mask, step)
            if step > 0:
                x = self._rotate_lmpdd(x)
                if anchor_latent is not None and anchor_mask is not None:
                    anchor_latent = self._rotate_lmpdd(anchor_latent)
                    anchor_mask = self._rotate_lmpdd(anchor_mask)
        return x

    def _validate_shape(self, shape: Sequence[int]) -> tuple[int, int, int, int]:
        if len(shape) != 4:
            raise ValueError("shape must be [B, C, H, W].")
        shape = tuple(int(value) for value in shape)
        if any(value <= 0 for value in shape):
            raise ValueError("shape values must be positive.")
        return shape

    def _validate_lmpdd_shape(self, shape: tuple[int, int, int, int]) -> None:
        if shape[0] != shape[2] or shape[0] != shape[3]:
            raise ValueError("L-MPDD sampling requires a cubic latent shape.")

    def _prepare_anchor(
        self,
        shape: tuple[int, int, int, int],
        *,
        anchor_latent: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (anchor_latent is None) != (anchor_mask is None):
            raise ValueError("anchor_latent and anchor_mask must be provided together.")
        if anchor_latent is None or anchor_mask is None:
            return None, None

        anchor_latent = anchor_latent.to(device=self.device)
        if anchor_latent.shape != torch.Size(shape):
            raise ValueError("anchor_latent must have the same shape as shape.")

        anchor_mask = anchor_mask.to(device=self.device, dtype=anchor_latent.dtype)
        try:
            anchor_mask = torch.broadcast_to(anchor_mask, anchor_latent.shape)
        except RuntimeError as exc:
            raise ValueError(
                "anchor_mask must be broadcastable to anchor_latent shape."
            ) from exc
        if anchor_mask.min().item() < 0.0 or anchor_mask.max().item() > 1.0:
            raise ValueError("anchor_mask values must be between 0 and 1.")

        return anchor_latent, anchor_mask

    def _blend_anchor(
        self,
        x: torch.Tensor,
        anchor_latent: torch.Tensor,
        anchor_mask: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if step == 0:
            anchor = anchor_latent
        else:
            t = torch.full(
                (anchor_latent.shape[0],),
                step - 1,
                dtype=torch.long,
                device=self.device,
            )
            anchor = self.ddpm.q_sample(anchor_latent, t)
        return x * (1.0 - anchor_mask) + anchor * anchor_mask

    def _rotate_lmpdd(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 2).transpose(3, 0).contiguous()
