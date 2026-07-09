import numpy as np
from PIL import Image


def crop_square(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("image must have shape [H, W].")
    if size <= 0:
        raise ValueError("size must be positive.")
    if image.shape[0] < size or image.shape[1] < size:
        raise ValueError(f"image is too small for a {size}x{size} crop.")

    max_y = image.shape[0] - size
    max_x = image.shape[1] - size
    y = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
    x = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
    return image[y : y + size, x : x + size]


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


def augment_patch(patch: np.ndarray) -> np.ndarray:
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1]:
        raise ValueError("patch must be a square 2D array.")

    transform = int(np.random.randint(0, 8))
    if transform >= 4:
        patch = np.fliplr(patch)
        transform -= 4
    return np.rot90(patch, transform)
