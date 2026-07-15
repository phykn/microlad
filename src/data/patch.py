import numpy as np
from PIL import Image

from ..misc import require_int


def crop(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("image must have shape [H, W].")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")

    require_int("size", size)

    if size <= 0:
        raise ValueError("size must be positive.")
    if image.shape[0] < size or image.shape[1] < size:
        raise ValueError(f"image is too small for a {size}x{size} crop.")

    max_y = image.shape[0] - size
    max_x = image.shape[1] - size
    y = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
    x = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
    return image[y : y + size, x : x + size]


def resize(patch: np.ndarray, size: int) -> np.ndarray:
    if patch.ndim != 2:
        raise ValueError("patch must have shape [H, W].")
    if patch.dtype != np.uint8:
        raise ValueError("patch must have dtype uint8.")

    require_int("size", size)

    if size <= 0:
        raise ValueError("size must be positive.")
    if patch.shape == (size, size):
        return patch.copy()

    image = Image.fromarray(patch)
    image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def augment(patch: np.ndarray) -> np.ndarray:
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1]:
        raise ValueError("patch must be a square 2D array.")
    if patch.dtype != np.uint8:
        raise ValueError("patch must have dtype uint8.")

    transform = int(np.random.randint(0, 8))
    if transform >= 4:
        patch = np.fliplr(patch)
        transform -= 4
    return np.rot90(patch, transform).copy()
