from collections.abc import Sequence

import torch

from src.guidance.conditioning.images import prepare_anchor_image
from src.guidance.conditioning.reconstruction import reconstruct_anchor_target
from src.guidance.conditioning.validation import validate_anchors
from src.guidance.conditioning.model import AnchorSlice
from src.tensors.validation import validate_floating_dtype


def prepare_anchor_targets(
    vae: torch.nn.Module,
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_shape: torch.Size,
    num_phases: int,
    segment: bool,
    device: torch.device,
    dtype: torch.dtype,
    tile_overlap: int = 0,
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
        target = reconstruct_anchor_target(
            vae,
            target,
            tile_overlap=tile_overlap,
        )
        targets[(anchor.axis, anchor.index)] = target

    return targets


def prepare_inference_module(module: torch.nn.Module) -> None:
    module.eval()

    for parameter in module.parameters():
        parameter.requires_grad_(False)
