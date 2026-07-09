import torch

from src.modeling.diffusion import DDPMProcess
from src.common.tensors.validation import validate_finite_tensor


def sds_loss(
    latent: torch.Tensor,
    model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    t_min: int,
    t_max: int,
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    spatial_weight: torch.Tensor | None = None,
    spatial_normalizer: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if latent.ndim != 4:
        raise ValueError("latent must have shape [B, C, H, W].")

    if any(size <= 0 for size in latent.shape):
        raise ValueError("latent dimensions must be positive.")

    validate_finite_tensor("latent", latent)
    _validate_timestep_range(ddpm, t_min, t_max)

    if t is None:
        t = torch.randint(t_min, t_max, (latent.shape[0],), device=latent.device)
    else:
        t = t.to(device=latent.device, dtype=torch.long)

    if t.ndim != 1 or t.shape[0] != latent.shape[0]:
        raise ValueError("t must have shape [B].")

    if t.min().item() < t_min or t.max().item() >= t_max:
        raise ValueError("t values must be inside the requested timestep range.")

    if noise is None:
        noise = torch.randn_like(latent)

    if noise.shape != latent.shape:
        raise ValueError("noise must have the same shape as latent.")

    validate_finite_tensor("noise", noise)
    noisy_latent = ddpm.q_sample(latent, t, noise=noise)

    with torch.no_grad():
        pred_noise = model(noisy_latent, t)

    if pred_noise.shape != latent.shape:
        raise ValueError("model output must have the same shape as latent.")

    validate_finite_tensor("model output", pred_noise)

    sigma = ddpm.sqrt_one_minus_alphas_cumprod[t].view(
        (latent.shape[0],) + (1,) * (latent.ndim - 1)
    )
    target = pred_noise.detach() - noise
    value = sigma.pow(2) * latent * target
    if spatial_weight is None:
        loss = value.mean()
    else:
        spatial_weight = spatial_weight.to(device=latent.device, dtype=latent.dtype)
        if spatial_weight.shape != latent.shape[-2:]:
            raise ValueError("spatial_weight must match latent spatial shape.")
        validate_finite_tensor("spatial_weight", spatial_weight)
        if torch.any(spatial_weight < 0):
            raise ValueError("spatial_weight must be non-negative.")

        if spatial_normalizer is None:
            normalizer = spatial_weight.sum()
        else:
            normalizer = torch.as_tensor(
                spatial_normalizer,
                device=latent.device,
                dtype=latent.dtype,
            )
        if not torch.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("spatial_normalizer must be positive and finite.")

        weight = spatial_weight.view(1, 1, *spatial_weight.shape)
        channel_count = latent.shape[0] * latent.shape[1]
        loss = (value * weight).sum() / (channel_count * normalizer)

    return loss, {"sds": loss.detach(), "t": t.detach()}


def _validate_timestep_range(ddpm: DDPMProcess, t_min: int, t_max: int) -> None:
    if t_min < 0 or t_max > ddpm.num_timesteps or t_min >= t_max:
        raise ValueError(
            "timestep range must satisfy 0 <= t_min < t_max <= num_timesteps."
        )
