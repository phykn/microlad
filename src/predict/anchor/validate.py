from collections.abc import Sequence

import numpy as np

from src.predict.types import AnchorSlice


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


def _slice_shape(volume_shape: Sequence[int], axis: int) -> tuple[int, int]:
    if axis == 0:
        return int(volume_shape[1]), int(volume_shape[2])

    if axis == 1:
        return int(volume_shape[0]), int(volume_shape[2])

    return int(volume_shape[0]), int(volume_shape[1])
