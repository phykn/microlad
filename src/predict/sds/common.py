from collections.abc import Sequence

import torch

from src.predict.anchor import prepare_anchor_image, validate_anchors
from src.predict.types import AnchorSlice
from src.predict.validation import validate_floating_dtype


def prepare_anchor_targets(
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_shape: torch.Size,
    num_phases: int,
    segment: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[tuple[int, int], torch.Tensor]:
    if not anchors:
        return {}

    validate_floating_dtype("dtype", dtype)
    validate_anchors(anchors, volume_shape)

    targets: dict[tuple[int, int], torch.Tensor] = {}

    for anchor in anchors:
        target = prepare_anchor_image(
            anchor.image,
            num_phases=num_phases,
            segment=segment,
        ).to(device=device, dtype=dtype)
        targets[(anchor.axis, anchor.index)] = target

    return targets


def prepare_inference_module(module: torch.nn.Module) -> None:
    module.eval()

    for parameter in module.parameters():
        parameter.requires_grad_(False)
