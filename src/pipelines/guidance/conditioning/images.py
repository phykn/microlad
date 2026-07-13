from collections.abc import Sequence

import numpy as np
import torch

from src.modeling.phases.quantization import MAX_UINT8_PHASES
from src.pipelines.data.segmentation import segment_otsu
from src.pipelines.guidance.conditioning.model import AnchorSlice, VolumeAnchor
from src.pipelines.guidance.conditioning.validation import (
    validate_anchor_intersections,
    validate_anchor_positions,
)
from src.validation import require_int


def prepare_anchor_image(
    image: np.ndarray,
    *,
    num_phases: int,
    segment: bool = False,
) -> torch.Tensor:
    phase = prepare_phase_image(
        image,
        num_phases=num_phases,
        segment=segment,
        name="anchor image",
    )
    return torch.from_numpy(phase.astype(np.float32, copy=True)).unsqueeze(0).unsqueeze(0)


def prepare_phase_image(
    image: np.ndarray,
    *,
    num_phases: int,
    segment: bool = False,
    name: str = "image",
) -> np.ndarray:
    require_int("num_phases", num_phases)
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if num_phases > MAX_UINT8_PHASES:
        raise ValueError(
            f"num_phases must be at most {MAX_UINT8_PHASES} for uint8 images."
        )
    if not isinstance(segment, bool):
        raise ValueError("segment must be a boolean.")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if image.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if image.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"{name} must contain numeric phase values.")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} values must be finite.")

    phase = segment_otsu(image, num_phases) if segment else image
    if not np.all(phase == np.rint(phase)):
        raise ValueError(f"{name} must contain integer phase values.")
    if phase.min() < 0 or phase.max() >= num_phases:
        raise ValueError(f"{name} must contain values from 0 to {num_phases - 1}.")
    return phase


def prepare_volume_anchors(
    anchors: Sequence[AnchorSlice] | None,
    *,
    volume_size: int,
    num_phases: int,
    segment: bool,
    device: torch.device,
    intersection_tolerance: float = 0.0,
) -> list[VolumeAnchor]:
    validate_anchor_positions(
        anchors or [],
        (volume_size, volume_size, volume_size),
    )
    prepared = []
    for anchor in anchors or []:
        labels = prepare_anchor_image(
            anchor.image,
            num_phases=num_phases,
            segment=segment,
        )[0, 0].to(device=device, dtype=torch.long)
        size = int(labels.shape[-1])
        if labels.shape != (size, size) or size > volume_size:
            raise ValueError("anchor image must be square and fit inside volume_size.")
        prepared.append(
            VolumeAnchor(
                image=labels,
                axis=int(anchor.axis),
                index=int(anchor.index),
                start=(volume_size - size) // 2,
            )
        )
    validate_anchor_intersections(
        prepared,
        tolerance=intersection_tolerance,
    )
    return prepared
