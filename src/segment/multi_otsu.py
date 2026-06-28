import numpy as np
from skimage.filters import threshold_multiotsu


def segment_multi_otsu(image: np.ndarray, num_phases: int) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape [H, W] or [D, H, W].")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")
    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    thresholds = threshold_multiotsu(image, classes=num_phases)
    return np.digitize(image, thresholds, right=True).astype(np.uint8)
