from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.modeling.diffusion import DDPMProcess
from src.pipelines.reconstruction.slices import (
    extract_slice,
    extract_slice_batch,
    replace_slice,
    replace_slice_batch,
    select_slice_batch,
)
from src.pipelines.guidance.preparation import (
    build_anchor_constraint_volume,
    freeze_inference,
)
from src.pipelines.guidance.physics.diffusivity import DiffusivitySolver
from src.pipelines.guidance.target_values import build_phase_target
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.guidance.evaluation import (
    _objective,
    _objective_batch,
)
from src.pipelines.reconstruction.volume import (
    decode_latent_with_probabilities,
    decode_latents_with_probabilities,
    decoded_labels,
)
from src.modeling.phases.representation import probabilities_to_calibrated_labels
from src.pipelines.guidance.validation import (
    _validate_inputs,
    _validate_contract,
    _validate_volume_inputs,
)

from src.common.tensors.validation import require_finite


def optimize_volume(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    steps: int,
    slice_steps: int,
    lr: float,
    t_min: int,
    t_max: int,
    num_phases: int,
    sds_batch_size: int = 1,
    slice_schedule: Sequence[tuple[int, int]] | None = None,
    anchors: Sequence[AnchorSlice] | None = None,
    anchor_segment: bool = False,
    sds_weight: float = 1.0,
    anchor_weight: float = 0.0,
    vf_targets: Mapping[int, float] | torch.Tensor | None = None,
    vf_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: DiffusivitySolver | None = None,
    diffusivity_weight: float = 0.0,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
    consensus_sweeps: bool = False,
    anchor_slab_radius: int = 0,
    anchor_slab_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_volume_inputs(
        volume,
        steps=steps,
        slice_steps=slice_steps,
        sds_batch_size=sds_batch_size,
        slice_schedule=slice_schedule,
        anchors=anchors,
        anchor_weight=anchor_weight,
    )
    _validate_contract(
        lr=lr,
        sds_weight=sds_weight,
        anchor_weight=anchor_weight,
        anchor_target=None,
        require_anchor_target=False,
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
    if not isinstance(consensus_sweeps, bool):
        raise ValueError("consensus_sweeps must be a boolean.")
    if not isinstance(anchor_slab_radius, int) or isinstance(anchor_slab_radius, bool):
        raise ValueError("anchor_slab_radius must be an integer.")
    if anchor_slab_radius < 0:
        raise ValueError("anchor_slab_radius must be non-negative.")
    if not 0.0 <= float(anchor_slab_weight) <= 1.0:
        raise ValueError("anchor_slab_weight must be between 0 and 1.")

    constraint_target, constraint_mask = build_anchor_constraint_volume(
        vae,
        anchors,
        volume_shape=volume.shape,
        num_phases=num_phases,
        segment=anchor_segment,
        device=volume.device,
        dtype=volume.dtype,
    )

    categorical = (
        getattr(vae, "num_phases", None) == num_phases
        and callable(getattr(vae, "decode_probs", None))
    )
    use_consensus = consensus_sweeps and categorical and steps > 0
    if use_consensus:
        _validate_consensus_schedule(
            volume,
            steps=steps,
            batch_size=sds_batch_size,
            schedule=slice_schedule,
        )

    updated = volume.clone().float()
    history: dict[str, list[torch.Tensor]] = {}

    if use_consensus:
        steps_per_sweep = 3 * int(volume.shape[0]) // sds_batch_size
        for sweep_start in range(0, steps, steps_per_sweep):
            snapshot = updated
            probability_sum = torch.zeros(
                (num_phases, *volume.shape),
                device=volume.device,
                dtype=volume.dtype,
            )
            vote_count = torch.zeros_like(volume)

            for step in range(sweep_start, sweep_start + steps_per_sweep):
                axis, indices = select_slice_batch(
                    snapshot,
                    step,
                    slice_schedule,
                    sds_batch_size,
                )
                targets, masks = _constraint_batch(
                    constraint_target,
                    constraint_mask,
                    axis=axis,
                    indices=indices,
                )
                proposal, step_stats, probabilities = _optimize_selected(
                    snapshot,
                    vae,
                    diffusion_model,
                    ddpm,
                    axis=axis,
                    indices=indices,
                    steps=slice_steps,
                    lr=lr,
                    t_min=t_min,
                    t_max=t_max,
                    num_phases=num_phases,
                    sds_weight=sds_weight,
                    anchor_targets=targets,
                    anchor_masks=masks,
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
                    return_probabilities=True,
                )
                del proposal
                _accumulate_probability_batch(
                    probability_sum,
                    vote_count,
                    probabilities,
                    axis=axis,
                    indices=indices,
                )
                _record_history(history, step_stats)

            probabilities = probability_sum / vote_count.clamp_min(1).unsqueeze(0)
            probabilities = _smooth_anchor_slabs(
                probabilities,
                anchors,
                radius=anchor_slab_radius,
                weight=anchor_slab_weight,
            )
            updated = probabilities_to_calibrated_labels(
                probabilities.unsqueeze(0),
                num_phases,
                target_fractions=_phase_fraction_target(
                    vf_targets,
                    num_phases=num_phases,
                    device=probabilities.device,
                    dtype=probabilities.dtype,
                ),
            )[0, 0].float()

        return updated, _finalize_history(history, steps, updated)

    for step in range(steps):
        axis, indices = select_slice_batch(
            updated,
            step,
            slice_schedule,
            sds_batch_size,
        )

        targets, masks = _constraint_batch(
            constraint_target,
            constraint_mask,
            axis=axis,
            indices=indices,
        )
        updated, step_stats = _optimize_selected(
            updated,
            vae,
            diffusion_model,
            ddpm,
            axis=axis,
            indices=indices,
            steps=slice_steps,
            lr=lr,
            t_min=t_min,
            t_max=t_max,
            num_phases=num_phases,
            sds_weight=sds_weight,
            anchor_targets=targets,
            anchor_masks=masks,
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
        )
        _record_history(history, step_stats)

    return updated, _finalize_history(history, steps, updated)


def _optimize_selected(
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
    anchor_masks: Sequence[torch.Tensor | None],
    anchor_weight: float,
    vf_targets: Mapping[int, float] | torch.Tensor | None,
    vf_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    tpc_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    sa_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: DiffusivitySolver | None,
    diffusivity_weight: float,
    temperature: float,
    sa_kernel_size: int,
    sa_sigma: float,
    return_probabilities: bool = False,
):
    if len(indices) == 1:
        target = anchor_targets[0]
        return optimize_slice(
            volume,
            vae,
            diffusion_model,
            ddpm,
            axis=axis,
            index=indices[0],
            steps=steps,
            lr=lr,
            t_min=t_min,
            t_max=t_max,
            num_phases=num_phases,
            sds_weight=sds_weight,
            anchor_target=target,
            anchor_mask=anchor_masks[0],
            anchor_weight=anchor_weight if target is not None else 0.0,
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
            return_probabilities=return_probabilities,
        )

    return _optimize_slice_batch(
        volume,
        vae,
        diffusion_model,
        ddpm,
        axis=axis,
        indices=indices,
        steps=steps,
        lr=lr,
        t_min=t_min,
        t_max=t_max,
        num_phases=num_phases,
        sds_weight=sds_weight,
        anchor_targets=anchor_targets,
        anchor_masks=anchor_masks,
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
        return_probabilities=return_probabilities,
    )


def _constraint_batch(
    target_volume: torch.Tensor,
    mask_volume: torch.Tensor,
    *,
    axis: int,
    indices: Sequence[int],
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    targets: list[torch.Tensor | None] = []
    masks: list[torch.Tensor | None] = []
    for index in indices:
        mask = extract_slice(mask_volume, axis, index)
        if bool((mask > 0).any().item()):
            targets.append(extract_slice(target_volume, axis, index))
            masks.append(mask)
        else:
            targets.append(None)
            masks.append(None)
    return targets, masks


def _validate_consensus_schedule(
    volume: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    schedule: Sequence[tuple[int, int]] | None,
) -> None:
    size = int(volume.shape[0])
    if schedule is None:
        raise ValueError("consensus_sweeps requires a balanced slice_schedule.")
    if size % batch_size != 0:
        raise ValueError("consensus_sweeps requires volume size divisible by batch size.")
    steps_per_sweep = 3 * size // batch_size
    if steps % steps_per_sweep != 0:
        raise ValueError("consensus_sweeps requires complete three-axis sweeps.")

    expected = {(axis, index) for axis in range(3) for index in range(size)}
    entries_per_sweep = steps_per_sweep * batch_size
    for start in range(0, steps * batch_size, entries_per_sweep):
        entries = schedule[start : start + entries_per_sweep]
        if len(entries) != entries_per_sweep or set(entries) != expected:
            raise ValueError(
                "consensus_sweeps requires every axis and slice exactly once per sweep."
            )


def _accumulate_probability_batch(
    probability_sum: torch.Tensor,
    vote_count: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    axis: int,
    indices: Sequence[int],
) -> None:
    index = torch.as_tensor(indices, device=probability_sum.device, dtype=torch.long)
    if axis == 0:
        probability_sum[:, index, :, :] += probabilities.permute(1, 0, 2, 3)
        vote_count[index, :, :] += 1
    elif axis == 1:
        probability_sum[:, :, index, :] += probabilities.permute(1, 2, 0, 3)
        vote_count[:, index, :] += 1
    else:
        probability_sum[:, :, :, index] += probabilities.permute(1, 2, 3, 0)
        vote_count[:, :, index] += 1


def _smooth_anchor_slabs(
    probabilities: torch.Tensor,
    anchors: Sequence[AnchorSlice] | None,
    *,
    radius: int,
    weight: float,
) -> torch.Tensor:
    if not anchors or radius == 0 or weight == 0.0:
        return probabilities

    refined = probabilities
    size = int(probabilities.shape[1])
    for anchor in anchors:
        source = refined.clone()
        updated = refined.clone()
        for offset in range(-radius, radius + 1):
            index = int(anchor.index) + offset
            if index < 0 or index >= size:
                continue
            neighbors = [
                _probability_plane(source, int(anchor.axis), neighbor)
                for neighbor in (index - 1, index, index + 1)
                if 0 <= neighbor < size
            ]
            smooth = torch.stack(neighbors).mean(dim=0)
            local_weight = weight * (radius + 1 - abs(offset)) / (radius + 1)
            current = _probability_plane(source, int(anchor.axis), index)
            _set_probability_plane(
                updated,
                int(anchor.axis),
                index,
                (1.0 - local_weight) * current + local_weight * smooth,
            )
        refined = updated
    return refined


def _probability_plane(values: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    if axis == 0:
        return values[:, index, :, :]
    if axis == 1:
        return values[:, :, index, :]
    return values[:, :, :, index]


def _set_probability_plane(
    values: torch.Tensor,
    axis: int,
    index: int,
    plane: torch.Tensor,
) -> None:
    if axis == 0:
        values[:, index, :, :] = plane
    elif axis == 1:
        values[:, :, index, :] = plane
    else:
        values[:, :, :, index] = plane


def _record_history(
    history: dict[str, list[torch.Tensor]],
    stats: dict[str, torch.Tensor],
) -> None:
    for key, value in stats.items():
        history.setdefault(key, []).append(value.detach())


def _phase_fraction_target(
    targets: Mapping[int, float] | torch.Tensor | None,
    *,
    num_phases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if targets is None:
        return None
    return build_phase_target(
        targets,
        num_phases=num_phases,
        device=device,
        dtype=dtype,
        label="fraction",
        require_sum_one=True,
    )


def _finalize_history(
    history: dict[str, list[torch.Tensor]],
    steps: int,
    volume: torch.Tensor,
) -> dict[str, torch.Tensor]:
    stats = {
        key: torch.stack(values).mean()
        for key, values in history.items()
        if values
    }
    stats["steps"] = torch.tensor(steps, device=volume.device)
    return stats


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
    diffusivity_solver: DiffusivitySolver | None = None,
    diffusivity_weight: float = 0.0,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
    return_probabilities: bool = False,
) -> (
    tuple[torch.Tensor, dict[str, torch.Tensor]]
    | tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]
):
    _validate_inputs(
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

    freeze_inference(vae)
    freeze_inference(diffusion_model)

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
        decoded, phase_probabilities = decode_latent_with_probabilities(
            vae,
            latent,
            num_phases=num_phases,
        )

        total, stats = _objective(
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
        decoded, phase_probabilities = decode_latent_with_probabilities(
            vae,
            latent,
            num_phases=num_phases,
        )
        if phase_probabilities is not None:
            decoded = probabilities_to_calibrated_labels(
                phase_probabilities.unsqueeze(0),
                num_phases,
            )[0, 0].float()
        replace_slice(updated, axis, index, decoded)

    if return_probabilities:
        probabilities = (
            _labels_to_probabilities(decoded.unsqueeze(0), num_phases)
            if phase_probabilities is None
            else phase_probabilities.unsqueeze(0).float()
        )
        return updated, stats, probabilities
    return updated, stats


def _optimize_slice_batch(
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
    diffusivity_solver: DiffusivitySolver | None,
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

    freeze_inference(vae)
    freeze_inference(diffusion_model)

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
        decoded, phase_probabilities = decode_latents_with_probabilities(
            vae,
            latent,
            num_phases=num_phases,
        )

        total, stats = _objective_batch(
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
        decoded, phase_probabilities = decode_latents_with_probabilities(
            vae,
            latent,
            num_phases=num_phases,
        )
        decoded = decoded_labels(
            decoded,
            phase_probabilities,
            num_phases=num_phases,
        )
        replace_slice_batch(updated, axis, indices, decoded)

    if return_probabilities:
        probabilities = (
            _labels_to_probabilities(decoded, num_phases)
            if phase_probabilities is None
            else phase_probabilities.float()
        )
        return updated, stats, probabilities
    return updated, stats


def _labels_to_probabilities(
    labels: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    indices = labels.round().clamp(0, num_phases - 1).to(torch.long)
    return F.one_hot(indices, num_classes=num_phases).permute(0, 3, 1, 2).float()
