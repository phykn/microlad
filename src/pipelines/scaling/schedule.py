from collections.abc import Sequence

import torch

from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.reconstruction.slices import (
    _validate_axis,
    _validate_index,
    _validate_indices,
    _validate_volume,
)
from src.pipelines.scaling.conditioning import anchor_positions
from src.validation import require_int


_AXIS_ORDERS = (
    (0, 1, 2),
    (2, 1, 0),
    (1, 0, 2),
    (2, 0, 1),
    (0, 2, 1),
    (1, 2, 0),
)


def select_slice(
    volume: torch.Tensor,
    step: int,
    schedule: Sequence[tuple[int, int]] | None,
) -> tuple[int, int]:
    _validate_volume(volume)
    require_int("step", step)
    if step < 0:
        raise ValueError("step must be non-negative.")

    if schedule is None:
        axis = int(torch.randint(0, 3, (), device=volume.device).item())
        index = int(
            torch.randint(0, volume.shape[axis], (), device=volume.device).item()
        )
    else:
        if step >= len(schedule):
            raise ValueError("slice_schedule must contain an entry for each step.")
        axis, index = schedule[step]

    try:
        _validate_axis(axis)
        _validate_index(volume, axis, index)
    except ValueError as exc:
        raise ValueError(f"slice_schedule {exc}") from exc

    return axis, index


def select_slice_batch(
    volume: torch.Tensor,
    step: int,
    schedule: Sequence[tuple[int, int]] | None,
    batch_size: int,
) -> tuple[int, list[int]]:
    _validate_volume(volume)
    require_int("step", step)
    require_int("batch_size", batch_size)
    if step < 0:
        raise ValueError("step must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if batch_size == 1:
        axis, index = select_slice(volume, step, schedule)
        return axis, [index]

    if schedule is not None:
        start = step * batch_size
        entries = schedule[start : start + batch_size]
        if len(entries) != batch_size:
            raise ValueError("slice_schedule must contain one entry per batched slice.")

        axes = [axis for axis, _ in entries]
        for axis in axes:
            try:
                _validate_axis(axis)
            except ValueError as exc:
                raise ValueError(f"slice_schedule {exc}") from exc
        if any(axis != axes[0] for axis in axes):
            raise ValueError("batched scale slices must use the same axis.")

        axis = axes[0]
        indices = [index for _, index in entries]
    else:
        valid_axes = [axis for axis in range(3) if volume.shape[axis] >= batch_size]
        if not valid_axes:
            raise ValueError("batch_size cannot exceed every volume axis length.")
        selected = int(
            torch.randint(0, len(valid_axes), (), device=volume.device).item()
        )
        axis = valid_axes[selected]
        indices = torch.randperm(
            volume.shape[axis],
            device=volume.device,
        )[:batch_size].tolist()

    try:
        _validate_indices(volume, axis, indices)
    except ValueError as exc:
        raise ValueError(f"slice_schedule {exc}") from exc
    if len(set(indices)) != len(indices):
        raise ValueError("batched scale slices must be unique.")

    return axis, indices


def build_anchor_schedule(
    anchors: Sequence[AnchorSlice] | None,
    *,
    steps: int,
    batch_size: int,
    volume_size: int,
    base_size: int,
    downsample_factor: int,
    device: torch.device,
) -> list[tuple[int, int]] | None:
    shifted = anchor_positions(
        anchors,
        volume_size=volume_size,
        base_size=base_size,
        downsample_factor=downsample_factor,
    )
    if not shifted or steps <= 0:
        return None
    if batch_size <= 0:
        raise ValueError("scale.batch_size must be positive.")
    if batch_size > volume_size:
        raise ValueError("scale.batch_size cannot exceed volume_size.")

    remaining = [(int(axis), int(index)) for axis, index in shifted]
    schedule: list[tuple[int, int]] = []

    for _ in range(steps):
        group: list[tuple[int, int]] = []
        used_indices: set[int] = set()

        if remaining:
            axis = remaining[0][0]
            next_remaining: list[tuple[int, int]] = []
            for entry_axis, entry_index in remaining:
                if (
                    entry_axis == axis
                    and len(group) < batch_size
                    and entry_index not in used_indices
                ):
                    group.append((entry_axis, entry_index))
                    used_indices.add(entry_index)
                else:
                    next_remaining.append((entry_axis, entry_index))
            remaining = next_remaining
        else:
            axis = int(torch.randint(0, 3, (), device=device).item())

        while len(group) < batch_size:
            candidates = torch.randperm(volume_size, device=device).tolist()
            index = next(
                (
                    int(candidate)
                    for candidate in candidates
                    if candidate not in used_indices
                ),
                None,
            )
            if index is None:
                raise ValueError("scale.batch_size cannot exceed volume_size.")
            group.append((axis, index))
            used_indices.add(index)

        schedule.extend(group)

    return schedule


def build_balanced_schedule(
    *,
    steps: int,
    batch_size: int,
    volume_size: int,
) -> list[tuple[int, int]] | None:
    if steps <= 0:
        return None
    if batch_size <= 0:
        raise ValueError("scale.batch_size must be positive.")
    if batch_size > volume_size:
        raise ValueError("scale.batch_size cannot exceed volume_size.")
    if volume_size % batch_size != 0:
        raise ValueError(
            "balanced scale guidance requires volume_size to be divisible by "
            "scale.batch_size."
        )

    batches_per_axis = volume_size // batch_size
    schedule: list[tuple[int, int]] = []
    for step in range(steps):
        sweep = step // (3 * batches_per_axis)
        within_sweep = step % (3 * batches_per_axis)
        axis_slot = within_sweep // batches_per_axis
        batch_slot = within_sweep % batches_per_axis
        axis = _AXIS_ORDERS[sweep % len(_AXIS_ORDERS)][axis_slot]
        start = batch_slot * batch_size
        schedule.extend((axis, index) for index in range(start, start + batch_size))

    return schedule
