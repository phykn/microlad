from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    with Image.open(Path(path)) as image:
        if image.mode == "P":
            return np.asarray(image.convert("L"), dtype=np.uint8)
        array = np.asarray(image)
        if array.ndim == 2:
            return _to_uint8_grayscale(array)
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _to_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()

    values = image.astype(np.float32)
    low = float(values.min())
    high = float(values.max())
    if 0.0 <= low and high <= 255.0:
        return image.astype(np.uint8)
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    values = (values - low) * (255.0 / (high - low))
    return values.astype(np.uint8)
