from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..data.segment import segment_otsu
from ..model import MAX_PHASES
from ..misc import require_int


@dataclass(frozen=True)
class AnchorSlice:
    image: np.ndarray
    axis: int
    index: int


@dataclass(frozen=True)
class _PreparedAnchor:
    image: torch.Tensor
    axis: int
    index: int
    start: int


def prepare_anchors(
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_size: int,
    num_phases: int,
    segment: bool,
    device: torch.device,
) -> list[_PreparedAnchor]:
    _check_positions(anchors or [], volume_size)
    prepared = []
    for anchor in anchors or []:
        labels = torch.from_numpy(
            _prepare_image(
                anchor.image,
                num_phases=num_phases,
                segment=segment,
            ).astype(np.int64, copy=True)
        ).to(device=device)
        size = int(labels.shape[-1])
        if labels.shape != (size, size) or size > volume_size:
            raise ValueError("anchor image must be square and fit inside volume_size.")
        prepared.append(
            _PreparedAnchor(
                image=labels,
                axis=int(anchor.axis),
                index=int(anchor.index),
                start=(volume_size - size) // 2,
            )
        )
    _check_crossings(prepared)
    return prepared


def build_constraints(
    volume_shape: tuple[int, int, int],
    anchors: Sequence[_PreparedAnchor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(volume_shape) != 3 or any(size <= 0 for size in volume_shape):
        raise ValueError("volume_shape must contain three positive values.")
    labels = torch.zeros(volume_shape, dtype=torch.long, device=device)
    mask = torch.zeros(volume_shape, dtype=torch.bool, device=device)

    for anchor in anchors:
        image = anchor.image.to(device=device, dtype=torch.long)
        size = int(image.shape[-1])
        if image.shape != (size, size):
            raise ValueError("volume anchor image must be square.")
        start = int(anchor.start)
        stop = start + size
        axis = int(anchor.axis)
        index = int(anchor.index)
        if axis not in (0, 1, 2):
            raise ValueError("volume anchor axis must be 0, 1, or 2.")
        if index < 0 or index >= volume_shape[axis]:
            raise ValueError("volume anchor index is outside the selected axis.")
        other_sizes = [volume_shape[value] for value in range(3) if value != axis]
        if start < 0 or any(stop > value for value in other_sizes):
            raise ValueError("volume anchor footprint is outside the volume.")

        label_view, mask_view = _get_views(
            labels,
            mask,
            axis=axis,
            index=index,
            start=start,
            stop=stop,
        )
        conflict = mask_view & (label_view != image)
        if bool(conflict.any().item()):
            raise ValueError(
                f"Conflicting anchor intersection at axis={axis}, index={index}."
            )
        label_view.copy_(torch.where(mask_view, label_view, image))
        mask_view.fill_(True)

    return labels, mask


def _prepare_image(
    image: np.ndarray,
    *,
    num_phases: int,
    segment: bool,
) -> np.ndarray:
    require_int("num_phases", num_phases)
    if num_phases < 2 or num_phases > MAX_PHASES:
        raise ValueError(f"num_phases must be between 2 and {MAX_PHASES}.")
    if not isinstance(segment, bool):
        raise ValueError("segment must be a boolean.")
    if not isinstance(image, np.ndarray):
        raise TypeError("anchor image must be a numpy array.")
    if image.ndim != 2:
        raise ValueError("anchor image must be 2D.")
    if image.size == 0:
        raise ValueError("anchor image must be non-empty.")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("anchor image must contain numeric phase values.")
    if not np.all(np.isfinite(image)):
        raise ValueError("anchor image values must be finite.")

    phase = segment_otsu(image, num_phases) if segment else image
    if not np.all(phase == np.rint(phase)):
        raise ValueError("anchor image must contain integer phase values.")
    if phase.min() < 0 or phase.max() >= num_phases:
        raise ValueError(
            f"anchor image must contain values from 0 to {num_phases - 1}."
        )
    return phase


def _check_positions(
    anchors: Sequence[AnchorSlice],
    volume_size: int,
) -> None:
    require_int("volume_size", volume_size)
    if volume_size <= 0:
        raise ValueError("volume_size must be positive.")

    seen: set[tuple[int, int]] = set()
    for anchor in anchors:
        require_int("axis", anchor.axis)
        require_int("index", anchor.index)
        if anchor.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2.")
        if anchor.index < 0 or anchor.index >= volume_size:
            raise ValueError("index is outside the selected axis.")
        key = (anchor.axis, anchor.index)
        if key in seen:
            raise ValueError(
                f"Duplicate anchor slice: axis={anchor.axis}, index={anchor.index}."
            )
        seen.add(key)


def _check_crossings(anchors: Sequence[_PreparedAnchor]) -> None:
    for first_index, first in enumerate(anchors):
        for second in anchors[first_index + 1 :]:
            if first.axis == second.axis:
                continue
            lines = _get_intersection(first, second)
            if lines is None:
                continue
            first_line, second_line = lines
            if bool((first_line != second_line).any().item()):
                raise ValueError(
                    "Conflicting anchor intersection: "
                    f"axis={first.axis}, index={first.index} and "
                    f"axis={second.axis}, index={second.index} disagree."
                )


def _get_intersection(
    first: _PreparedAnchor,
    second: _PreparedAnchor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if first.axis > second.axis:
        lines = _get_intersection(second, first)
        if lines is None:
            return None
        second_line, first_line = lines
        return first_line, second_line

    first_size = int(first.image.shape[-1])
    second_size = int(second.image.shape[-1])
    overlap_start = max(first.start, second.start)
    overlap_stop = min(first.start + first_size, second.start + second_size)
    if overlap_start >= overlap_stop:
        return None

    first_begin = overlap_start - first.start
    first_end = overlap_stop - first.start
    second_begin = overlap_start - second.start
    second_end = overlap_stop - second.start

    def get_local(anchor: _PreparedAnchor, absolute: int) -> int | None:
        index = absolute - anchor.start
        return index if 0 <= index < int(anchor.image.shape[-1]) else None

    axes = (first.axis, second.axis)
    if axes == (0, 1):
        first_row = get_local(first, second.index)
        second_row = get_local(second, first.index)
        if first_row is None or second_row is None:
            return None
        return (
            first.image[first_row, first_begin:first_end],
            second.image[second_row, second_begin:second_end],
        )
    if axes == (0, 2):
        first_col = get_local(first, second.index)
        second_row = get_local(second, first.index)
        if first_col is None or second_row is None:
            return None
        return (
            first.image[first_begin:first_end, first_col],
            second.image[second_row, second_begin:second_end],
        )
    if axes == (1, 2):
        first_col = get_local(first, second.index)
        second_col = get_local(second, first.index)
        if first_col is None or second_col is None:
            return None
        return (
            first.image[first_begin:first_end, first_col],
            second.image[second_begin:second_end, second_col],
        )
    raise ValueError("anchor axes must be different values from 0, 1, and 2.")


def _get_views(
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    axis: int,
    index: int,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if axis == 0:
        return labels[index, start:stop, start:stop], mask[
            index, start:stop, start:stop
        ]
    if axis == 1:
        return labels[start:stop, index, start:stop], mask[
            start:stop, index, start:stop
        ]
    return labels[start:stop, start:stop, index], mask[start:stop, start:stop, index]
