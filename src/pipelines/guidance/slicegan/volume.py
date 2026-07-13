from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from src.modeling.phases import probabilities_to_calibrated_labels
from src.modeling.slicegan import SliceGANGenerator
from src.pipelines.guidance.slicegan.anchors import PreparedAnchor
from src.pipelines.guidance.slicegan.render import render_generator_tiled
from src.pipelines.reconstruction.volume import decode_volume_probs
from src.pipelines.scaling.decoding import decode_large_volume_probabilities

if TYPE_CHECKING:
    from src.app.api.options import SliceGANRenderConfig


def decode_probabilities(
    vae: torch.nn.Module,
    latent: torch.Tensor,
    *,
    num_phases: int,
    batch_size: int = 16,
) -> torch.Tensor:
    if latent.shape[-3:] == (
        int(vae.latent_size),
        int(vae.latent_size),
        int(vae.latent_size),
    ):
        return decode_volume_probs(vae, latent, num_phases=num_phases)
    return decode_large_volume_probabilities(
        vae,
        latent,
        tile_overlap=max(1, int(vae.latent_size) // 4),
        num_phases=num_phases,
        batch_size=batch_size,
    )


def calibrate(
    probabilities: torch.Tensor,
    anchors: Sequence[PreparedAnchor],
    *,
    target_fraction: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    selected = probabilities.argmax(dim=1, keepdim=True)
    mask = torch.zeros_like(selected, dtype=torch.bool)
    for anchor in anchors:
        size = int(anchor.labels.shape[-1])
        start = anchor.start
        stop = start + size
        if anchor.axis == 0:
            mask[:, :, anchor.index, start:stop, start:stop] = True
        elif anchor.axis == 1:
            mask[:, :, start:stop, anchor.index, start:stop] = True
        else:
            mask[:, :, start:stop, start:stop, anchor.index] = True
    return probabilities_to_calibrated_labels(
        probabilities,
        num_phases,
        target_fractions=target_fraction,
        fixed_labels=selected,
        fixed_mask=mask,
    )[0, 0]


def render_latent(
    generator: SliceGANGenerator,
    noise: torch.Tensor,
    *,
    render: "SliceGANRenderConfig",
) -> torch.Tensor:
    generator.eval()
    if max(map(int, noise.shape[-3:])) <= render.core_noise_size:
        return generator(noise)
    return render_generator_tiled(
        generator,
        noise,
        core_noise_size=render.core_noise_size,
        halo_noise_size=render.halo_noise_size,
        output_device=noise.device,
    )


def decode_frozen(vae: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    parameters = tuple(vae.parameters())
    states = tuple(parameter.requires_grad for parameter in parameters)
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        return vae.decode_probs(latent)
    finally:
        for parameter, state in zip(parameters, states, strict=True):
            parameter.requires_grad_(state)
