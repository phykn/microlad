import numpy as np
import torch

from src.segment import segment_multi_otsu


def prepare_anchor_image(
    image: np.ndarray,
    *,
    num_phases: int,
    segment: bool = False,
) -> torch.Tensor:
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array.")
    if image.ndim != 2:
        raise ValueError("anchor image must be 2D.")

    phase = segment_multi_otsu(image, num_phases) if segment else image
    if phase.min() < 0 or phase.max() >= num_phases:
        raise ValueError(
            f"anchor image must contain values from 0 to {num_phases - 1}."
        )

    scaled = phase.astype(np.float32) / (num_phases - 1) * 2.0 - 1.0
    return torch.from_numpy(scaled.copy()).unsqueeze(0).unsqueeze(0).float()
