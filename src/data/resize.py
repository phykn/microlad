import numpy as np
from PIL import Image


def resize_patch(patch: np.ndarray, size: int) -> np.ndarray:
    if patch.ndim != 2:
        raise ValueError("patch must have shape [H, W].")

    if size <= 0:
        raise ValueError("size must be positive.")

    if patch.shape == (size, size):
        return patch

    image = Image.fromarray(patch)
    image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)
