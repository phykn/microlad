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


def load_phase_image(path: str | Path) -> np.ndarray:
    with Image.open(Path(path)) as image:
        array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError("phase image must be a 2D label image.")

    return _to_uint8_phase_labels(array)


def _to_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()

    values = image.astype(np.float32)

    if not np.isfinite(values).all():
        raise ValueError("image values must be finite.")

    low = float(values.min())
    high = float(values.max())

    if np.issubdtype(image.dtype, np.floating) and 0.0 <= low and high <= 1.0:
        rounded = np.round(values)

        if np.allclose(values, rounded):
            return rounded.astype(np.uint8)

        return (values * 255.0).astype(np.uint8)

    if 0.0 <= low and high <= 255.0:
        return image.astype(np.uint8)

    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)

    values = (values - low) * (255.0 / (high - low))
    return values.astype(np.uint8)


def _to_uint8_phase_labels(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        raise ValueError("phase image must be non-empty.")

    if np.issubdtype(image.dtype, np.floating):
        return _float_phase_labels_to_uint8(image)

    values = image.astype(np.int64)
    low = int(values.min())
    high = int(values.max())

    if low < 0 or high > int(np.iinfo(np.uint8).max):
        raise ValueError("phase image labels must fit in uint8.")

    return values.astype(np.uint8)


def _float_phase_labels_to_uint8(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)

    if not np.isfinite(values).all():
        raise ValueError("phase image labels must be finite.")

    rounded = np.rint(values)

    if not np.allclose(values, rounded):
        raise ValueError("phase image labels must be integer values.")

    low = float(rounded.min())
    high = float(rounded.max())

    if low < 0.0 or high > float(np.iinfo(np.uint8).max):
        raise ValueError("phase image labels must fit in uint8.")

    return rounded.astype(np.uint8)
