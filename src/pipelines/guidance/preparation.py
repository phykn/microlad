from collections.abc import Sequence

import torch

from src.pipelines.guidance.conditioning.images import prepare_anchor_image
from src.pipelines.guidance.conditioning.reconstruction import reconstruct_target
from src.pipelines.guidance.conditioning.validation import validate_anchors
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.common.tensors.validation import require_float


def build_anchor_targets(
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

    require_float("dtype", dtype)
    validate_anchors(anchors, volume_shape)

    targets: dict[tuple[int, int], torch.Tensor] = {}

    for anchor in anchors:
        target = prepare_anchor_image(
            anchor.image,
            num_phases=num_phases,
            segment=segment,
        ).to(device=device, dtype=dtype)
        target = reconstruct_target(
            vae,
            target,
            tile_overlap=tile_overlap,
        )
        targets[(anchor.axis, anchor.index)] = target

    return targets


def freeze_inference(module: torch.nn.Module) -> None:
    module.eval()

    for parameter in module.parameters():
        parameter.requires_grad_(False)
