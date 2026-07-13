from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.modeling.slicegan import (
    NOISE_CHANNELS,
    SliceGANCritic,
    SliceGANGenerator,
    generator_loss,
)
from src.modeling.slicegan.sampling import sample_slices
from src.pipelines.guidance.slicegan.anchors import PreparedAnchor
from src.pipelines.guidance.slicegan.volume import decode_frozen, render_latent

if TYPE_CHECKING:
    from src.app.api.options import SliceGANConditionConfig, SliceGANRenderConfig


def resolve_target_fraction(
    anchors: Sequence[PreparedAnchor],
    *,
    target_fraction: torch.Tensor | None,
    num_phases: int,
    device: torch.device,
) -> torch.Tensor:
    if target_fraction is not None:
        return torch.as_tensor(target_fraction, device=device, dtype=torch.float32)
    counts = torch.zeros(num_phases, device=device)
    total = 0
    for anchor in anchors:
        counts += torch.bincount(anchor.labels.flatten(), minlength=num_phases)
        total += anchor.labels.numel()
    return counts.float() / float(total)


def fit_noise(
    generator: SliceGANGenerator,
    critic: SliceGANCritic,
    vae: torch.nn.Module,
    anchors: Sequence[PreparedAnchor],
    *,
    noise_size: int,
    steps: int,
    config: "SliceGANConditionConfig",
    device: torch.device,
) -> tuple[torch.nn.Parameter, dict[str, torch.Tensor]]:
    generator.eval()
    critic.eval()
    _set_requires_grad(generator, False)
    _set_requires_grad(critic, False)
    noise = _select_noise(
        generator,
        vae,
        anchors,
        noise_size=noise_size,
        candidates=config.candidates,
        device=device,
    )
    optimizer = torch.optim.Adam([noise], lr=config.noise_lr)
    pixel = shape = noise_prior = torch.zeros((), device=device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        latent = generator(noise)
        pixel, shape = _anchor_loss(vae, latent, anchors)
        fake = sample_slices(
            latent,
            count=6,
            crop_size=int(vae.latent_size),
        )
        prior = generator_loss(critic(fake))
        noise_prior = _noise_distribution_loss(noise)
        loss = pixel + 2.0 * shape + config.critic_weight * prior + 0.05 * noise_prior
        loss.backward()
        torch.nn.utils.clip_grad_norm_([noise], max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            noise.clamp_(-3.5, 3.5)
    return noise, {
        "slicegan_noise_steps": torch.tensor(steps, device=device),
        "slicegan_noise_pixel_loss": pixel.detach(),
        "slicegan_noise_shape_loss": shape.detach(),
        "slicegan_noise_prior": noise_prior.detach(),
    }


def tune_condition(
    generator: SliceGANGenerator,
    critic: SliceGANCritic,
    vae: torch.nn.Module,
    noise: torch.nn.Parameter,
    anchors: Sequence[PreparedAnchor],
    target_fraction: torch.Tensor,
    *,
    target_transition: torch.Tensor,
    factor: int,
    steps: int,
    config: "SliceGANConditionConfig",
    render: "SliceGANRenderConfig",
    optimizer_betas: tuple[float, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    generator.eval()
    critic.eval()
    _set_requires_grad(generator, True)
    _set_requires_grad(critic, False)
    with torch.no_grad():
        baseline = generator(noise).detach()
    preserve = latent_preservation_weights(
        baseline.shape[-3:],
        anchors,
        factor=factor,
        sigma=config.influence_sigma,
        device=baseline.device,
        dtype=baseline.dtype,
    )
    optimizer = torch.optim.Adam(
        [
            {"params": generator.parameters(), "lr": config.generator_lr},
            {"params": [noise], "lr": config.tune_noise_lr},
        ],
        betas=optimizer_betas,
    )
    pixel = shape = phase = transition = preservation = baseline.new_zeros(())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        latent = generator(noise)
        pixel, shape = _anchor_loss(vae, latent, anchors)
        pair = _decode_adjacent_pair(vae, latent)
        phase = F.mse_loss(pair.mean(dim=(0, 2, 3)), target_fraction)
        actual_transition = (1.0 - (pair[0] * pair[1]).sum(dim=0)).mean()
        transition = F.mse_loss(actual_transition, target_transition)
        prior = generator_loss(
            critic(
                sample_slices(
                    latent,
                    count=6,
                    crop_size=int(vae.latent_size),
                )
            )
        )
        preservation = ((latent - baseline).square() * preserve).mean()
        loss = (
            pixel
            + 2.0 * shape
            + config.phase_weight * phase
            + config.transition_weight * transition
            + config.tune_critic_weight * prior
            + config.preserve_weight * preservation
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([noise], max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            noise.clamp_(-3.5, 3.5)

    generator.eval()
    with torch.no_grad():
        latent = render_latent(generator, noise, render=render)[0]
    return latent, {
        "slicegan_tune_steps": torch.tensor(steps, device=latent.device),
        "slicegan_tune_pixel_loss": pixel.detach(),
        "slicegan_tune_shape_loss": shape.detach(),
        "slicegan_tune_phase_loss": phase.detach(),
        "slicegan_tune_transition_loss": transition.detach(),
        "slicegan_tune_preservation_loss": preservation.detach(),
    }


def latent_preservation_weights(
    spatial_shape: Sequence[int],
    anchors: Sequence[PreparedAnchor],
    *,
    factor: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    shape = tuple(map(int, spatial_shape))
    coordinates = torch.meshgrid(
        *[torch.arange(size, device=device, dtype=dtype) for size in shape],
        indexing="ij",
    )
    influence = torch.zeros(shape, device=device, dtype=dtype)
    latent_sigma = max(float(sigma) / factor, 1e-6)
    for anchor in anchors:
        index = float(anchor.index) / factor
        start = float(anchor.start) / factor
        stop = start + float(anchor.labels.shape[-1]) / factor - 1.0
        distance = (coordinates[anchor.axis] - index).square()
        for axis in range(3):
            if axis == anchor.axis:
                continue
            distance += (
                F.relu(start - coordinates[axis])
                + F.relu(coordinates[axis] - stop)
            ).square()
        influence = torch.maximum(
            influence,
            torch.exp(-0.5 * distance / latent_sigma**2),
        )
    return (1.0 - influence).reshape(1, 1, *shape)


def _select_noise(
    generator: SliceGANGenerator,
    vae: torch.nn.Module,
    anchors: Sequence[PreparedAnchor],
    *,
    noise_size: int,
    candidates: int,
    device: torch.device,
) -> torch.nn.Parameter:
    best = None
    best_score = float("inf")
    with torch.no_grad():
        for _ in range(candidates):
            noise = torch.randn(
                1,
                NOISE_CHANNELS,
                noise_size,
                noise_size,
                noise_size,
                device=device,
            )
            latent = generator(noise)
            pixel, shape = _anchor_loss(vae, latent, anchors)
            score = float((pixel + shape).item())
            if score < best_score:
                best_score = score
                best = noise.detach().clone()
    if best is None:
        raise RuntimeError("failed to initialize SliceGAN noise.")
    return torch.nn.Parameter(best)


def _anchor_loss(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    anchors: Sequence[PreparedAnchor],
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = []
    shapes = []
    for anchor in anchors:
        for patch in anchor.patches:
            plane = latent[0].select(patch.axis + 1, patch.latent_index)
            size = int(vae.latent_size)
            crop = plane[
                :,
                patch.latent_row : patch.latent_row + size,
                patch.latent_col : patch.latent_col + size,
            ]
            if crop.shape[-2:] != (size, size):
                raise ValueError("anchor latent patch falls outside generated volume.")
            probabilities = decode_frozen(vae, crop.unsqueeze(0))[0]
            pixels.append(
                F.nll_loss(
                    probabilities.clamp_min(1e-8).log().unsqueeze(0),
                    patch.labels.unsqueeze(0),
                )
            )
            shapes.append(_multiscale_shape_loss(probabilities, patch.probabilities))
    return torch.stack(pixels).mean(), torch.stack(shapes).mean()


def _decode_adjacent_pair(
    vae: torch.nn.Module,
    latent: torch.Tensor,
) -> torch.Tensor:
    axis = int(torch.randint(0, 3, (), device=latent.device).item())
    length = int(latent.shape[axis + 2])
    index = int(torch.randint(0, length - 1, (), device=latent.device).item())
    planes = [latent[0].select(axis + 1, index + offset) for offset in (0, 1)]
    size = int(vae.latent_size)
    max_row = int(planes[0].shape[-2]) - size
    max_col = int(planes[0].shape[-1]) - size
    row = int(torch.randint(0, max_row + 1, (), device=latent.device).item())
    col = int(torch.randint(0, max_col + 1, (), device=latent.device).item())
    batch = torch.stack(
        [plane[:, row : row + size, col : col + size] for plane in planes]
    )
    return decode_frozen(vae, batch)


def _multiscale_shape_loss(
    actual: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for scale in (2, 4, 8):
        if scale <= min(actual.shape[-2:]):
            losses.append(
                F.mse_loss(
                    F.avg_pool2d(actual.unsqueeze(0), scale, stride=scale),
                    F.avg_pool2d(target.unsqueeze(0), scale, stride=scale),
                )
            )
    return torch.stack(losses).mean()


def _noise_distribution_loss(noise: torch.Tensor) -> torch.Tensor:
    tail = F.relu(noise.abs() - 3.0).square().mean()
    return (
        noise.mean().square()
        + (noise.std(unbiased=False) - 1.0).square()
        + tail
    )


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
