import numpy as np
import torch

from src.phases.quantization import MAX_UINT8_PHASES
from src.phases.segmentation import segment_multi_otsu


def prepare_anchor_image(
    image: np.ndarray,
    *,
    num_phases: int,
    segment: bool = False,
) -> torch.Tensor:
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if num_phases > MAX_UINT8_PHASES:
        raise ValueError(
            f"num_phases must be at most {MAX_UINT8_PHASES} for uint8 images."
        )

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array.")

    if image.ndim != 2:
        raise ValueError("anchor image must be 2D.")

    phase = segment_multi_otsu(image, num_phases) if segment else image
    _validate_phase_image(phase, num_phases)

    return torch.from_numpy(phase.astype(np.float32, copy=True)).unsqueeze(0).unsqueeze(0)


def _validate_phase_image(phase: np.ndarray, num_phases: int) -> None:
    if not np.issubdtype(phase.dtype, np.number):
        raise TypeError("anchor image must contain numeric phase values.")

    if not np.all(np.isfinite(phase)):
        raise ValueError("anchor image values must be finite.")

    if not np.all(phase == np.rint(phase)):
        raise ValueError("anchor image must contain integer phase values.")

    if phase.min() < 0 or phase.max() >= num_phases:
        raise ValueError(
            f"anchor image must contain values from 0 to {num_phases - 1}."
        )
