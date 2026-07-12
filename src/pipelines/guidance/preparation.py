from collections.abc import Sequence

import torch
import torch.nn.functional as F

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


def build_anchor_constraint_volume(
    vae: torch.nn.Module,
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_shape: torch.Size,
    num_phases: int,
    segment: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project every anchor plane into one shared 3D categorical constraint field."""
    categorical = (
        getattr(vae, "num_phases", None) == num_phases
        and callable(getattr(vae, "decode_probs", None))
    )
    if categorical:
        validate_anchors(anchors or [], volume_shape)
        targets = {
            (int(anchor.axis), int(anchor.index)): prepare_anchor_image(
                anchor.image,
                num_phases=num_phases,
                segment=segment,
            ).to(device=device, dtype=dtype)
            for anchor in anchors or []
        }
    else:
        targets = build_anchor_targets(
            vae,
            anchors,
            volume_shape=volume_shape,
            num_phases=num_phases,
            segment=segment,
            device=device,
            dtype=dtype,
        )
    votes = torch.zeros(
        (num_phases, *volume_shape),
        device=device,
        dtype=dtype,
    )
    counts = torch.zeros(volume_shape, device=device, dtype=dtype)

    for (axis, index), target in targets.items():
        labels = target[0, 0].round().clamp(0, num_phases - 1).to(torch.long)
        one_hot = F.one_hot(labels, num_classes=num_phases).permute(2, 0, 1)
        _add_constraint_plane(votes, counts, one_hot.to(dtype), axis, index)

    mask = (counts > 0).to(dtype)
    target_volume = votes.argmax(dim=0).to(dtype)
    return target_volume, mask


def _add_constraint_plane(
    votes: torch.Tensor,
    counts: torch.Tensor,
    one_hot: torch.Tensor,
    axis: int,
    index: int,
) -> None:
    if axis == 0:
        votes[:, index, :, :] += one_hot
        counts[index, :, :] += 1
    elif axis == 1:
        votes[:, :, index, :] += one_hot
        counts[:, index, :] += 1
    else:
        votes[:, :, :, index] += one_hot
        counts[:, :, index] += 1


def freeze_inference(module: torch.nn.Module) -> None:
    module.eval()

    for parameter in module.parameters():
        parameter.requires_grad_(False)
