from collections.abc import Sequence

import torch

from src.predict.anchor import prepare_anchor_image, validate_anchors
from src.predict.types import AnchorSlice


def apply_anchor_slices(
    volume: torch.Tensor,
    anchors: Sequence[AnchorSlice],
    *,
    num_phases: int,
    segment: bool = False,
) -> torch.Tensor:
    if volume.ndim != 3:
        raise ValueError("volume must have shape [D, H, W].")
    validate_anchors(anchors, volume.shape)

    out = volume.clone()
    for anchor in anchors:
        image = prepare_anchor_image(
            anchor.image,
            num_phases=num_phases,
            segment=segment,
        )[0, 0].to(device=out.device, dtype=out.dtype)
        _write_slice(out, image, axis=anchor.axis, index=anchor.index)
    return out


def _write_slice(
    volume: torch.Tensor,
    image: torch.Tensor,
    *,
    axis: int,
    index: int,
) -> None:
    if axis == 0:
        volume[index, :, :] = image
    elif axis == 1:
        volume[:, index, :] = image
    else:
        volume[:, :, index] = image
