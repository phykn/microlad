from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure

from src.pipelines.guidance.conditioning.model import VolumeAnchor
from src.pipelines.guidance.metrics.runs import compute_run_profile
from src.pipelines.guidance.metrics.topology import compute_euler_density
from src.pipelines.reconstruction.slices import extract_slice
from src.pipelines.guidance.metrics.slices import volume_slices


def evaluate_phase_volume(
    volume: torch.Tensor,
    *,
    num_phases: int,
    references: torch.Tensor | None = None,
    target_fraction: torch.Tensor | None = None,
    anchors: Sequence[VolumeAnchor] = (),
    run_lengths: Sequence[int] = (2, 4, 8, 16),
) -> dict[str, torch.Tensor]:
    labels, probabilities = _prepare_volume(volume, num_phases=num_phases)
    lengths = tuple(
        length for length in run_lengths if length <= min(map(int, labels.shape))
    )
    if not lengths:
        lengths = (min(map(int, labels.shape)),)

    transition_rates = []
    lag3_change_rates = []
    exact_repeat_rates = []
    near_repeat_rates = []
    max_repeat_streaks = []
    max_adjacent_similarities = []
    global_boundary_jumps = []
    axis_euler_density = []
    for axis in range(3):
        length = int(labels.shape[axis])
        before = labels.narrow(axis, 0, length - 1)
        after = labels.narrow(axis, 1, length - 1)
        changed = (before != after).float()
        reduce_axes = tuple(dimension for dimension in range(3) if dimension != axis)
        boundary_profile = changed.mean(dim=reduce_axes)
        transition_rates.append(boundary_profile.mean())
        global_boundary_jumps.append(
            torch.abs(boundary_profile[1:] - boundary_profile[:-1]).max()
            if boundary_profile.numel() > 1
            else boundary_profile.new_zeros(())
        )
        same_slices = (before == after).movedim(axis, 0).reshape(length - 1, -1)
        similarities = same_slices.float().mean(dim=1)
        repeated = same_slices.all(dim=1)
        exact_repeat_rates.append(repeated.float().mean())
        near_repeat_rates.append((similarities >= 0.99).float().mean())
        max_adjacent_similarities.append(similarities.max())
        streak = 0
        longest = 0
        for is_repeated in repeated.tolist():
            streak = streak + 1 if is_repeated else 0
            longest = max(longest, streak)
        max_repeat_streaks.append(similarities.new_tensor(float(longest)))
        lag = min(3, length - 1)
        lag3_change_rates.append(
            (
                labels.narrow(axis, 0, length - lag)
                != labels.narrow(axis, lag, length - lag)
            )
            .float()
            .mean()
        )
        axis_euler_density.append(
            compute_euler_density(
                volume_slices(probabilities, axis, num_phases=num_phases)
            ).mean(dim=0)
        )

    measured_fraction = probabilities.mean(dim=(0, 2, 3, 4))
    stats = {
        "phase_fraction": measured_fraction,
        "axis_transition_rate": torch.stack(transition_rates),
        "axis_lag3_change_rate": torch.stack(lag3_change_rates),
        "axis_exact_repeat_rate": torch.stack(exact_repeat_rates),
        "axis_near_repeat_rate": torch.stack(near_repeat_rates),
        "axis_max_repeat_streak": torch.stack(max_repeat_streaks),
        "axis_max_adjacent_similarity": torch.stack(max_adjacent_similarities),
        "axis_global_boundary_jump": torch.stack(global_boundary_jumps),
        "axis_run_profile": compute_run_profile(probabilities, lengths=lengths),
        "axis_euler_density": torch.stack(axis_euler_density),
    }
    stats.update(_topology_stats(labels, num_phases=num_phases))

    if target_fraction is not None:
        target = torch.as_tensor(
            target_fraction,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        if target.shape != (num_phases,):
            raise ValueError("target_fraction must have shape [num_phases].")
        stats["target_phase_fraction"] = target
        stats["phase_fraction_error"] = measured_fraction - target

    if references is not None:
        reference = references.to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        if reference.ndim != 4 or reference.shape[1] != num_phases:
            raise ValueError("references must have shape [B, P, H, W].")
        reference_lengths = tuple(
            length
            for length in lengths
            if length <= min(map(int, reference.shape[-2:]))
        )
        if not reference_lengths:
            raise ValueError("run_lengths must fit inside the reference images.")
        axis_run = stats["axis_run_profile"][:, :, : len(reference_lengths)]
        target_run = compute_run_profile(
            reference,
            lengths=reference_lengths,
        ).mean(dim=0)
        target_euler = compute_euler_density(reference).mean(dim=0)
        axis_euler = stats["axis_euler_density"]
        stats.update(
            {
                "target_run_profile": target_run,
                "axis_run_profile_mae": torch.mean(
                    torch.abs(axis_run - target_run.unsqueeze(0)),
                    dim=(1, 2),
                ),
                "target_euler_density": target_euler,
                "axis_euler_mae": torch.mean(
                    torch.abs(axis_euler - target_euler.unsqueeze(0)),
                    dim=1,
                ),
            }
        )

    if anchors:
        mismatches = []
        phase_mismatches = []
        for anchor in anchors:
            image = anchor.image.to(device=labels.device, dtype=labels.dtype)
            size = int(image.shape[-1])
            start = int(anchor.start)
            actual = extract_slice(labels, int(anchor.axis), int(anchor.index))
            actual = actual[start : start + size, start : start + size]
            if actual.shape != image.shape:
                raise ValueError(
                    "anchor image must fit inside the selected volume slice."
                )
            mismatches.append((actual != image).float().mean())
            phase_mismatches.append(
                torch.stack(
                    [
                        (actual[image == phase] != phase).float().mean()
                        if bool((image == phase).any().item())
                        else actual.new_zeros((), dtype=torch.float32)
                        for phase in range(num_phases)
                    ]
                )
            )
        anchor_mismatches = torch.stack(mismatches)
        anchor_phase_mismatches = torch.stack(phase_mismatches)
        stats.update(
            {
                "anchor_mismatches": anchor_mismatches,
                "anchor_mismatch": anchor_mismatches.mean(),
                "anchor_max_mismatch": anchor_mismatches.max(),
                "anchor_phase_mismatches": anchor_phase_mismatches,
                "anchor_max_phase_mismatch": anchor_phase_mismatches.max(),
            }
        )

    return stats


def _topology_stats(
    labels: torch.Tensor,
    *,
    num_phases: int,
) -> dict[str, torch.Tensor]:
    array = labels.detach().cpu().numpy()
    components = []
    euler = []
    percolation = []
    for phase in range(num_phases):
        mask = array == phase
        component_labels = measure.label(mask, connectivity=1)
        components.append(float(component_labels.max()))
        euler.append(float(measure.euler_number(mask, connectivity=1)) / mask.size)
        phase_percolation = []
        for axis in range(3):
            first = np.unique(np.take(component_labels, 0, axis=axis))
            last = np.unique(np.take(component_labels, -1, axis=axis))
            spanning = np.intersect1d(first[first > 0], last[last > 0])
            phase_percolation.append(float(bool(spanning.size)))
        percolation.append(phase_percolation)
    return {
        "component_count": torch.tensor(
            components,
            device=labels.device,
            dtype=torch.float32,
        ),
        "euler_3d_density": torch.tensor(
            euler,
            device=labels.device,
            dtype=torch.float32,
        ),
        "phase_axis_percolation": torch.tensor(
            percolation,
            device=labels.device,
            dtype=torch.float32,
        ),
    }


def _prepare_volume(
    volume: torch.Tensor,
    *,
    num_phases: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")
    if any(int(size) < 2 for size in volume.shape):
        raise ValueError("volume dimensions must contain at least two voxels.")
    if not torch.isfinite(volume).all():
        raise ValueError("volume labels must be finite.")
    if volume.is_floating_point() and not torch.equal(volume, volume.round()):
        raise ValueError("volume labels must contain integer phase values.")
    labels = volume.to(dtype=torch.long)
    if labels.min().item() < 0 or labels.max().item() >= num_phases:
        raise ValueError("volume labels must be inside the phase range.")
    probabilities = (
        F.one_hot(labels, num_classes=num_phases).movedim(-1, 0).unsqueeze(0).float()
    )
    return labels, probabilities
