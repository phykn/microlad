from collections.abc import Mapping, Sequence

import torch

from src.modeling.diffusion import DDPMProcess
from src.pipelines.reconstruction.slices import extract_slice
from src.pipelines.guidance.conditioning.prepare import (
    build_anchor_constraint_volume,
)
from src.pipelines.guidance.metrics.conductance import ConductanceSolver
from src.pipelines.guidance.sds.schedule import select_slice_batch
from src.pipelines.guidance.metrics.targets import build_phase_target
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.guidance.sds.slice import optimize_slice, optimize_slice_batch
from src.modeling.phases.calibration import probabilities_to_calibrated_labels
from src.pipelines.guidance.sds.validation import (
    validate_loss,
    validate_volume,
)


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
    batch_size: int = 1,
    slice_schedule: Sequence[tuple[int, int]] | None = None,
    anchors: Sequence[AnchorSlice] | None = None,
    segment_anchors: bool = False,
    sds_weight: float = 1.0,
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
    consensus_sweeps: bool = False,
    anchor_slab_radius: int = 0,
    anchor_slab_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    validate_volume(
        volume,
        steps=steps,
        slice_steps=slice_steps,
        batch_size=batch_size,
        slice_schedule=slice_schedule,
        anchors=anchors,
        anchor_weight=anchor_weight,
    )
    validate_loss(
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
        segment=segment_anchors,
        device=volume.device,
        dtype=volume.dtype,
    )

    categorical = getattr(vae, "num_phases", None) == num_phases and callable(
        getattr(vae, "decode_probs", None)
    )
    use_consensus = consensus_sweeps and categorical and steps > 0
    if use_consensus:
        _validate_consensus_schedule(
            volume,
            steps=steps,
            batch_size=batch_size,
            schedule=slice_schedule,
        )

    updated = volume.clone().float()
    history: dict[str, list[torch.Tensor]] = {}

    if use_consensus:
        steps_per_sweep = 3 * int(volume.shape[0]) // batch_size
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
                    batch_size,
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
                fixed_labels=probabilities.argmax(dim=0)
                .unsqueeze(0)
                .unsqueeze(0),
                fixed_mask=(constraint_mask > 0).unsqueeze(0).unsqueeze(0),
            )[0, 0].float()

        return updated, _finalize_history(history, steps, updated)

    for step in range(steps):
        axis, indices = select_slice_batch(
            updated,
            step,
            slice_schedule,
            batch_size,
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
    diffusivity_solver: ConductanceSolver | None,
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

    return optimize_slice_batch(
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
        raise ValueError(
            "consensus_sweeps requires volume size divisible by batch size."
        )
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
        f"history_{key}": torch.stack(values).mean(dim=0)
        for key, values in history.items()
        if values
    }
    stats["steps"] = torch.tensor(steps, device=volume.device)
    return stats
