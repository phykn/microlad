from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.modeling.diffusion import DDPMProcess
from src.modeling.inference import freeze
from src.modeling.phases.calibration import probabilities_to_calibrated_labels
from src.pipelines.guidance.metrics.conductance import ConductanceSolver
from src.pipelines.guidance.sds.loss import batch_loss, slice_loss
from src.pipelines.guidance.sds.validation import validate_slice
from src.pipelines.reconstruction.slices import (
    extract_slice,
    extract_slice_batch,
    replace_slice,
    replace_slice_batch,
)
from src.pipelines.reconstruction.volume import decode_latent, decode_latents
from src.validation import require_finite


def optimize_slice(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    axis: int,
    index: int,
    steps: int,
    lr: float,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_weight: float = 1.0,
    anchor_target: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
    vf_targets: Mapping[int, float] | torch.Tensor | None = None,
    vf_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: ConductanceSolver | None = None,
    diffusivity_weight: float = 0.0,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
    return_probabilities: bool = False,
) -> (
    tuple[torch.Tensor, dict[str, torch.Tensor]]
    | tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]
):
    validate_slice(
        volume,
        vae,
        axis=axis,
        index=index,
        steps=steps,
        lr=lr,
        sds_weight=sds_weight,
        anchor_weight=anchor_weight,
        anchor_target=anchor_target,
        vf_weight=vf_weight,
        vf_targets=vf_targets,
        tpc_weight=tpc_weight,
        tpc_targets=tpc_targets,
        sa_weight=sa_weight,
        sa_targets=sa_targets,
        diffusivity_weight=diffusivity_weight,
        diffusivity_targets=diffusivity_targets,
        diffusivity_solver=diffusivity_solver,
    )

    updated = volume.clone().float()
    if steps == 0:
        if return_probabilities:
            image = extract_slice(updated, axis, index)
            probabilities = _labels_to_probabilities(image.unsqueeze(0), num_phases)
            return updated, {}, probabilities
        return updated, {}

    freeze(vae)
    freeze(diffusion_model)

    image = extract_slice(updated, axis, index).view(
        1,
        1,
        int(vae.image_size),
        int(vae.image_size),
    )
    mu, _ = vae.encode(image)

    if mu.ndim != 4:
        raise ValueError("vae.encode must return latent with shape [B, C, H, W].")

    require_finite("latent", mu)

    latent = mu.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)

    stats: dict[str, torch.Tensor] = {}

    for _ in range(steps):
        optimizer.zero_grad()
        decoded, phase_probabilities = decode_latent(
            vae,
            latent,
            num_phases=num_phases,
        )

        total, stats = slice_loss(
            latent,
            decoded,
            diffusion_model,
            ddpm,
            t_min=t_min,
            t_max=t_max,
            num_phases=num_phases,
            sds_weight=sds_weight,
            anchor_target=anchor_target,
            anchor_weight=anchor_weight,
            vf_targets=vf_targets,
            vf_weight=vf_weight,
            tpc_targets=tpc_targets,
            tpc_weight=tpc_weight,
            sa_targets=sa_targets,
            sa_weight=sa_weight,
            diffusivity_targets=diffusivity_targets,
            diffusivity_solver=diffusivity_solver,
            diffusivity_weight=diffusivity_weight,
            temperature=temperature,
            sa_kernel_size=sa_kernel_size,
            sa_sigma=sa_sigma,
            phase_probabilities=phase_probabilities,
            anchor_mask=anchor_mask,
        )
        total.backward()
        optimizer.step()

    with torch.no_grad():
        decoded, phase_probabilities = decode_latent(
            vae,
            latent,
            num_phases=num_phases,
        )
        fixed_mask = None
        fixed_labels = None
        if anchor_target is not None:
            mask = (
                torch.ones_like(decoded, dtype=torch.bool)
                if anchor_mask is None
                else anchor_mask > 0
            )
            fixed_mask = mask.unsqueeze(0).unsqueeze(0)
            fixed_labels = phase_probabilities.argmax(dim=0).unsqueeze(0).unsqueeze(0)
        decoded = probabilities_to_calibrated_labels(
            phase_probabilities.unsqueeze(0),
            num_phases,
            fixed_labels=fixed_labels,
            fixed_mask=fixed_mask,
        )[0, 0].float()
        replace_slice(updated, axis, index, decoded)

    if return_probabilities:
        return updated, stats, phase_probabilities.unsqueeze(0).float()
    return updated, stats


def optimize_slice_batch(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    axis: int,
    indices: Sequence[int],
    steps: int,
    lr: float,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_weight: float,
    anchor_targets: Sequence[torch.Tensor | None],
    anchor_masks: Sequence[torch.Tensor | None] | None,
    anchor_weight: float,
    vf_targets: Mapping[int, float] | torch.Tensor | None,
    vf_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    tpc_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    sa_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: ConductanceSolver | None,
    diffusivity_weight: float,
    temperature: float,
    sa_kernel_size: int,
    sa_sigma: float,
    return_probabilities: bool = False,
) -> (
    tuple[torch.Tensor, dict[str, torch.Tensor]]
    | tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]
):
    updated = volume.clone().float()
    if steps == 0:
        if return_probabilities:
            images = extract_slice_batch(updated, axis, indices)
            return updated, {}, _labels_to_probabilities(images, num_phases)
        return updated, {}

    freeze(vae)
    freeze(diffusion_model)

    images = extract_slice_batch(updated, axis, indices)
    image_size = int(vae.image_size)
    if images.shape[-2:] != torch.Size([image_size, image_size]):
        raise ValueError("selected slice shape must match vae.image_size.")

    latent, _ = vae.encode(images.view(len(indices), 1, image_size, image_size))

    if latent.ndim != 4 or latent.shape[0] != len(indices):
        raise ValueError("vae.encode must return latent with shape [B, C, H, W].")

    require_finite("latent", latent)

    latent = latent.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)

    stats: dict[str, torch.Tensor] = {}

    for _ in range(steps):
        optimizer.zero_grad()
        decoded, phase_probabilities = decode_latents(
            vae,
            latent,
            num_phases=num_phases,
        )

        total, stats = batch_loss(
            latent,
            decoded,
            diffusion_model,
            ddpm,
            t_min=t_min,
            t_max=t_max,
            num_phases=num_phases,
            sds_weight=sds_weight,
            anchor_targets=anchor_targets,
            anchor_weight=anchor_weight,
            vf_targets=vf_targets,
            vf_weight=vf_weight,
            tpc_targets=tpc_targets,
            tpc_weight=tpc_weight,
            sa_targets=sa_targets,
            sa_weight=sa_weight,
            diffusivity_targets=diffusivity_targets,
            diffusivity_solver=diffusivity_solver,
            diffusivity_weight=diffusivity_weight,
            temperature=temperature,
            sa_kernel_size=sa_kernel_size,
            sa_sigma=sa_sigma,
            phase_probabilities=phase_probabilities,
            anchor_masks=anchor_masks,
        )
        total.backward()
        optimizer.step()

    with torch.no_grad():
        decoded, phase_probabilities = decode_latents(
            vae,
            latent,
            num_phases=num_phases,
        )
        fixed_mask = torch.zeros(
            phase_probabilities.shape[0],
            1,
            *phase_probabilities.shape[-2:],
            device=phase_probabilities.device,
            dtype=torch.bool,
        )
        masks = anchor_masks or [None] * len(anchor_targets)
        for batch_index, (target, mask) in enumerate(
            zip(anchor_targets, masks, strict=True)
        ):
            if target is not None:
                fixed_mask[batch_index, 0] = (
                    torch.ones_like(target, dtype=torch.bool)
                    if mask is None
                    else mask > 0
                )
        decoded = probabilities_to_calibrated_labels(
            phase_probabilities,
            num_phases,
            fixed_labels=phase_probabilities.argmax(dim=1, keepdim=True),
            fixed_mask=fixed_mask,
        )[:, 0].float()
        replace_slice_batch(updated, axis, indices, decoded)

    if return_probabilities:
        return updated, stats, phase_probabilities.float()
    return updated, stats


def _labels_to_probabilities(
    labels: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    indices = labels.round().clamp(0, num_phases - 1).to(torch.long)
    return F.one_hot(indices, num_classes=num_phases).permute(0, 3, 1, 2).float()

