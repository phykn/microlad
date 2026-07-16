import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddpm import DDPMProcess


def compute_loss(
    model: nn.Module,
    ddpm: DDPMProcess,
    clean: torch.Tensor,
    fractions: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    axis_condition: torch.Tensor | None = None,
    anchor_image: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    anchor_loss_weight: float = 0.0,
    anchor_phase_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if clean.ndim != 4:
        raise ValueError("clean_image must have shape [B, C, H, W].")

    if any(size <= 0 for size in clean.shape):
        raise ValueError("clean_image dimensions must be positive.")

    if t is None:
        t = ddpm.sample_steps(clean.shape[0], device=clean.device)
    elif t.device != clean.device:
        raise ValueError("t must be on the same device as clean_image.")

    if noise is None:
        noise = torch.randn_like(clean)

    if noise.shape != clean.shape:
        raise ValueError("noise must have the same shape as clean_image.")
    if anchor_loss_weight < 0.0:
        raise ValueError("anchor_loss_weight must be non-negative.")
    if not math.isfinite(anchor_phase_loss_weight) or anchor_phase_loss_weight < 0.0:
        raise ValueError(
            "anchor_phase_loss_weight must be finite and non-negative."
        )
    _validate_anchor_condition(anchor_image, anchor_mask, clean)

    noisy = ddpm.add_noise(clean, t, noise=noise)
    if anchor_image is not None and anchor_mask is not None:
        if axis_condition is not None:
            _validate_axis_condition(axis_condition, clean)
        pred_noise = model(
            noisy,
            t,
            fractions,
            axis_condition,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
    elif axis_condition is None:
        pred_noise = (
            model(noisy, t) if fractions is None else model(noisy, t, fractions)
        )
    else:
        _validate_axis_condition(axis_condition, clean)
        pred_noise = model(noisy, t, fractions, axis_condition)

    if pred_noise.shape != noise.shape:
        raise ValueError("model output must have the same shape as noise.")

    squared_error = F.mse_loss(pred_noise, noise, reduction="none")
    per_pixel = squared_error.mean(dim=1, keepdim=True)
    noise_loss = per_pixel.mean()
    loss = noise_loss
    parts = {"noise": noise_loss.detach()}
    if anchor_mask is not None:
        parts["anchor_coverage"] = anchor_mask.to(torch.float32).mean().detach()
    if (
        anchor_mask is not None
        and anchor_loss_weight > 0.0
    ):
        release_step = _model_attribute(model, "anchor_release_step", 0)
        active = (t >= release_step).to(dtype=per_pixel.dtype)
        band = F.max_pool2d(
            anchor_mask.to(dtype=per_pixel.dtype),
            kernel_size=7,
            stride=1,
            padding=3,
        )
        band = band * active[:, None, None, None]
        area = band.flatten(start_dim=1).sum(dim=1)
        selected = area > 0
        if bool(selected.any().item()):
            region = (per_pixel * band).flatten(start_dim=1).sum(dim=1)
            anchor_loss = (region[selected] / area[selected]).mean()
            loss = loss + anchor_loss_weight * anchor_loss
            parts["anchor"] = anchor_loss.detach()
    if (
        anchor_image is not None
        and anchor_mask is not None
        and anchor_phase_loss_weight > 0.0
    ):
        release_step = _model_attribute(model, "anchor_release_step", 0)
        active = (t >= release_step).to(dtype=pred_noise.dtype)
        sigma = ddpm.sqrt_one_minus_alphas_cumprod[t].view(
            (t.shape[0],) + (1,) * (noisy.ndim - 1)
        )
        # This is alpha_t * x_0.  Its channel argmax is the x_0 phase without
        # the unstable division by alpha_t at high-noise timesteps.
        phase_logits = noisy - sigma * pred_noise
        phase_error = F.cross_entropy(
            phase_logits,
            anchor_image.argmax(dim=1),
            reduction="none",
        )
        mask = anchor_mask[:, 0].to(dtype=phase_error.dtype)
        mask = mask * active[:, None, None]
        area = mask.flatten(start_dim=1).sum(dim=1)
        selected = area > 0
        if bool(selected.any().item()):
            region = (phase_error * mask).flatten(start_dim=1).sum(dim=1)
            anchor_phase_loss = (region[selected] / area[selected]).mean()
            loss = loss + anchor_phase_loss_weight * anchor_phase_loss
            parts["anchor_phase"] = anchor_phase_loss.detach()
    if axis_condition is not None:
        per_sample = squared_error.flatten(start_dim=1).mean(dim=1)
        for axis in range(3):
            selected = axis_condition == axis
            if bool(selected.any().item()):
                parts[f"axis_{axis}"] = per_sample[selected].mean().detach()
    return loss, parts


def _validate_axis_condition(
    axis_condition: torch.Tensor,
    clean: torch.Tensor,
) -> None:
    if not isinstance(axis_condition, torch.Tensor):
        raise TypeError("axis_condition must be a tensor.")
    if axis_condition.shape != (clean.shape[0],):
        raise ValueError("axis_condition must have shape [B].")
    if axis_condition.dtype != torch.long:
        raise TypeError("axis_condition must have dtype torch.long.")
    if axis_condition.device != clean.device:
        raise ValueError("axis_condition must be on the same device as clean_image.")
    if bool(((axis_condition < 0) | (axis_condition >= 3)).any().item()):
        raise ValueError("axis_condition values must be in the range 0 to 2.")


class DiffusionLoss(nn.Module):
    def __init__(
        self,
        ddpm: DDPMProcess,
        *,
        anchor_loss_weight: float = 0.0,
        anchor_phase_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if anchor_loss_weight < 0.0:
            raise ValueError("anchor_loss_weight must be non-negative.")
        if not math.isfinite(anchor_phase_loss_weight) or anchor_phase_loss_weight < 0.0:
            raise ValueError(
                "anchor_phase_loss_weight must be finite and non-negative."
            )
        self.ddpm = ddpm
        self.anchor_loss_weight = float(anchor_loss_weight)
        self.anchor_phase_loss_weight = float(anchor_phase_loss_weight)

    def forward(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        fractions: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        axis_condition: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return compute_loss(
            model,
            self.ddpm,
            clean,
            fractions=fractions,
            t=t,
            noise=noise,
            axis_condition=axis_condition,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
            anchor_loss_weight=self.anchor_loss_weight,
            anchor_phase_loss_weight=self.anchor_phase_loss_weight,
        )


def _validate_anchor_condition(
    anchor_image: torch.Tensor | None,
    anchor_mask: torch.Tensor | None,
    clean: torch.Tensor,
) -> None:
    if (anchor_image is None) != (anchor_mask is None):
        raise ValueError("anchor_image and anchor_mask must be provided together.")
    if anchor_image is None or anchor_mask is None:
        return
    if anchor_image.shape != clean.shape:
        raise ValueError("anchor_image must have the same shape as clean_image.")
    if anchor_mask.shape != (clean.shape[0], 1, *clean.shape[-2:]):
        raise ValueError("anchor_mask must have shape [B, 1, H, W].")
    if anchor_image.device != clean.device or anchor_mask.device != clean.device:
        raise ValueError("anchor inputs must be on the same device as clean_image.")
    if not torch.isfinite(anchor_image).all():
        raise ValueError("anchor_image must be finite.")


def _model_attribute(model: nn.Module, name: str, default: int) -> int:
    value = getattr(model, name, None)
    if value is None and hasattr(model, "module"):
        value = getattr(model.module, name, None)
    return int(default if value is None else value)
