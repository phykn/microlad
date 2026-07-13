from collections.abc import Sequence

import torch
from tqdm import tqdm

from src.modeling.diffusion.process import DDPMProcess
from src.validation import require_finite


class DiffusionSampler:
    def __init__(
        self,
        model: torch.nn.Module,
        ddpm: DDPMProcess,
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
        self.model.eval()

        x = torch.randn(shape, device=self.device)
        anchor_latent, anchor_mask = self._prepare_anchor(
            shape,
            dtype=x.dtype,
            anchor_latent=anchor_latent,
            anchor_mask=anchor_mask,
        )

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
        axis_consensus: bool = False,
        progress: bool = False,
    ) -> torch.Tensor:
        shape = self._validate_shape(shape)
        if shape[0] != shape[2] or shape[0] != shape[3]:
            raise ValueError("L-MPDD sampling requires a cubic latent shape.")
        if not isinstance(axis_consensus, bool):
            raise ValueError("axis_consensus must be a boolean.")
        if not isinstance(progress, bool):
            raise ValueError("progress must be a boolean.")
        self.model.eval()

        x = torch.randn(shape, device=self.device)
        anchor_latent, anchor_mask = self._prepare_anchor(
            shape,
            dtype=x.dtype,
            anchor_latent=anchor_latent,
            anchor_mask=anchor_mask,
        )

        batch_size = shape[0]
        if axis_consensus:
            steps = tqdm(
                range(self.ddpm.num_timesteps - 1, -1, -1),
                total=self.ddpm.num_timesteps,
                desc="L-MPDD",
                disable=not progress,
            )
            for step in steps:
                t = torch.full(
                    (batch_size,),
                    step,
                    dtype=torch.long,
                    device=self.device,
                )
                pred_noise = self._predict_lmpdd_noise(x, t)
                x = self.ddpm.p_sample_from_noise(x, t, pred_noise)
                if anchor_latent is not None and anchor_mask is not None:
                    x = self._blend_anchor(x, anchor_latent, anchor_mask, step)
            return x

        rotations = 0
        steps = tqdm(
            range(self.ddpm.num_timesteps - 1, -1, -1),
            total=self.ddpm.num_timesteps,
            desc="L-MPDD",
            disable=not progress,
        )
        for step in steps:
            t = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            x = self.ddpm.p_sample(self.model, x, t)

            if anchor_latent is not None and anchor_mask is not None:
                x = self._blend_anchor(x, anchor_latent, anchor_mask, step)

            if step > 0:
                x = self._rotate_lmpdd(x)
                rotations += 1

                if anchor_latent is not None and anchor_mask is not None:
                    anchor_latent = self._rotate_lmpdd(anchor_latent)
                    anchor_mask = self._rotate_lmpdd(anchor_mask)

        for _ in range((3 - rotations % 3) % 3):
            x = self._rotate_lmpdd(x)

        return x

    def _validate_shape(self, shape: Sequence[int]) -> tuple[int, int, int, int]:
        if len(shape) != 4:
            raise ValueError("shape must be [B, C, H, W].")

        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in shape
        ):
            raise ValueError("shape values must be integers.")

        shape = tuple(shape)
        if any(value <= 0 for value in shape):
            raise ValueError("shape values must be positive.")

        return shape

    def _prepare_anchor(
        self,
        shape: tuple[int, int, int, int],
        *,
        dtype: torch.dtype,
        anchor_latent: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (anchor_latent is None) != (anchor_mask is None):
            raise ValueError("anchor_latent and anchor_mask must be provided together.")

        if anchor_latent is None or anchor_mask is None:
            return None, None

        anchor_latent = anchor_latent.to(device=self.device, dtype=dtype)
        if anchor_latent.shape != torch.Size(shape):
            raise ValueError("anchor_latent must have the same shape as shape.")

        require_finite("anchor_latent", anchor_latent)

        anchor_mask = anchor_mask.to(device=self.device, dtype=dtype)
        try:
            anchor_mask = torch.broadcast_to(anchor_mask, anchor_latent.shape)
        except RuntimeError as exc:
            raise ValueError(
                "anchor_mask must be broadcastable to anchor_latent shape."
            ) from exc

        require_finite("anchor_mask", anchor_mask)

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

    def _predict_lmpdd_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        oriented = x
        canonical_predictions = []
        for orientation in range(3):
            prediction = self.model(oriented, t)
            if prediction.shape != oriented.shape:
                raise ValueError("model output must have the same shape as x.")
            for _ in range((3 - orientation) % 3):
                prediction = self._rotate_lmpdd(prediction)
            canonical_predictions.append(prediction)
            oriented = self._rotate_lmpdd(oriented)
        return torch.stack(canonical_predictions).mean(dim=0)
