import numpy as np
from skimage.filters import threshold_multiotsu

from src.validation import require_int


def segment_otsu(image: np.ndarray, num_phases: int) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape [H, W] or [D, H, W].")

    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")

    require_int("num_phases", num_phases)

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if num_phases > int(np.iinfo(np.uint8).max) + 1:
        raise ValueError("num_phases must be at most 256 for uint8 images.")

    if image.size == 0:
        raise ValueError("image must be non-empty.")

    if np.unique(image).size < num_phases:
        raise ValueError(
            f"image must contain at least {num_phases} distinct intensity values."
        )

    thresholds = threshold_multiotsu(image, classes=num_phases)
    return np.digitize(image, thresholds, right=True).astype(np.uint8)
