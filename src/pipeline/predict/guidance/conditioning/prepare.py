from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.pipeline.predict.guidance.conditioning.images import prepare_anchor_image
from src.pipeline.predict.guidance.conditioning.reconstruction import reconstruct_target
from src.pipeline.predict.guidance.conditioning.validation import validate_anchors
from src.pipeline.predict.guidance.conditioning.model import AnchorSlice, VolumeAnchor
from src.validation import require_float


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
        count_plane = counts.select(axis, index)
        vote_plane = votes.select(axis + 1, index)
        conflict = (count_plane > 0) & (vote_plane.argmax(dim=0) != labels)
        if bool(conflict.any().item()):
            raise ValueError(
                f"Conflicting anchor intersection at axis={axis}, index={index}."
            )
        _add_constraint_plane(votes, counts, one_hot.to(dtype), axis, index)

    mask = (counts > 0).to(dtype)
    target_volume = votes.argmax(dim=0).to(dtype)
    return target_volume, mask


def build_volume_anchor_mask(
    volume_shape: tuple[int, int, int],
    anchors: Sequence[VolumeAnchor],
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(
        1,
        1,
        *volume_shape,
        dtype=torch.bool,
        device=device,
    )
    for anchor in anchors:
        length = int(anchor.image.shape[-1])
        start = int(anchor.start)
        stop = start + length
        if anchor.axis == 0:
            mask[:, :, anchor.index, start:stop, start:stop] = True
        elif anchor.axis == 1:
            mask[:, :, start:stop, anchor.index, start:stop] = True
        else:
            mask[:, :, start:stop, start:stop, anchor.index] = True
    return mask


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


