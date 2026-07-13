from collections.abc import Sequence

import numpy as np
import torch

from src.pipelines.guidance.conditioning.model import (
    AnchorSlice,
    VolumeAnchor,
)


def validate_anchor(anchor: AnchorSlice, volume_shape: Sequence[int]) -> None:
    if len(volume_shape) != 3:
        raise ValueError("volume_shape must have three values: [D, H, W].")

    if anchor.axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    if anchor.index < 0 or anchor.index >= volume_shape[anchor.axis]:
        raise ValueError("index is outside the selected axis.")

    if not isinstance(anchor.image, np.ndarray):
        raise TypeError("anchor image must be a numpy array.")

    if anchor.image.ndim != 2:
        raise ValueError("anchor image must be 2D.")

    expected_shape = _slice_shape(volume_shape, anchor.axis)
    if anchor.image.shape != expected_shape:
        raise ValueError(
            f"anchor image shape must be {expected_shape} for axis {anchor.axis}."
        )


def validate_anchors(
    anchors: Sequence[AnchorSlice],
    volume_shape: Sequence[int],
) -> None:
    seen: set[tuple[int, int]] = set()

    for anchor in anchors:
        validate_anchor(anchor, volume_shape)

        key = (anchor.axis, anchor.index)

        if key in seen:
            raise ValueError(
                f"Duplicate anchor slice: axis={anchor.axis}, index={anchor.index}."
            )

        seen.add(key)


def validate_anchor_intersections(
    anchors: Sequence[VolumeAnchor],
    *,
    tolerance: float,
) -> None:
    """Ensure different-axis categorical constraints agree where they cross."""

    if not np.isfinite(tolerance) or tolerance < 0.0 or tolerance > 1.0:
        raise ValueError("anchor intersection tolerance must be between 0 and 1.")
    seen: set[tuple[int, int]] = set()
    for anchor in anchors:
        key = (int(anchor.axis), int(anchor.index))
        if key in seen:
            raise ValueError(
                f"Duplicate anchor slice: axis={anchor.axis}, index={anchor.index}."
            )
        seen.add(key)

    for first_index, first in enumerate(anchors):
        for second in anchors[first_index + 1 :]:
            if first.axis == second.axis:
                continue
            lines = _categorical_anchor_intersection_lines(first, second)
            if lines is None:
                continue
            first_line, second_line = lines
            mismatch = (first_line != second_line).float().mean()
            if float(mismatch.item()) > tolerance:
                raise ValueError(
                    "Conflicting anchor intersection: "
                    f"axis={first.axis}, index={first.index} and "
                    f"axis={second.axis}, index={second.index} disagree at "
                    f"{float(mismatch.item()):.1%} of their intersection."
                )


def _categorical_anchor_intersection_lines(
    first: VolumeAnchor,
    second: VolumeAnchor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if first.axis > second.axis:
        lines = _categorical_anchor_intersection_lines(second, first)
        if lines is None:
            return None
        second_line, first_line = lines
        return first_line, second_line

    first_size = int(first.image.shape[-1])
    second_size = int(second.image.shape[-1])

    def local_index(anchor: VolumeAnchor, absolute_index: int) -> int | None:
        index = int(absolute_index) - int(anchor.start)
        return index if 0 <= index < int(anchor.image.shape[-1]) else None

    overlap_start = max(int(first.start), int(second.start))
    overlap_stop = min(
        int(first.start) + first_size,
        int(second.start) + second_size,
    )
    if overlap_start >= overlap_stop:
        return None
    first_begin = overlap_start - int(first.start)
    first_end = overlap_stop - int(first.start)
    second_begin = overlap_start - int(second.start)
    second_end = overlap_stop - int(second.start)

    axes = (first.axis, second.axis)
    if axes == (0, 1):
        first_row = local_index(first, second.index)
        second_row = local_index(second, first.index)
        if first_row is None or second_row is None:
            return None
        return (
            first.image[first_row, first_begin:first_end],
            second.image[second_row, second_begin:second_end],
        )
    if axes == (0, 2):
        first_col = local_index(first, second.index)
        second_row = local_index(second, first.index)
        if first_col is None or second_row is None:
            return None
        return (
            first.image[first_begin:first_end, first_col],
            second.image[second_row, second_begin:second_end],
        )
    if axes == (1, 2):
        first_col = local_index(first, second.index)
        second_col = local_index(second, first.index)
        if first_col is None or second_col is None:
            return None
        return (
            first.image[first_begin:first_end, first_col],
            second.image[second_begin:second_end, second_col],
        )
    raise ValueError("anchor axes must be different values from 0, 1, and 2.")


def _slice_shape(volume_shape: Sequence[int], axis: int) -> tuple[int, int]:
    if axis == 0:
        return int(volume_shape[1]), int(volume_shape[2])

    if axis == 1:
        return int(volume_shape[0]), int(volume_shape[2])

    return int(volume_shape[0]), int(volume_shape[1])
