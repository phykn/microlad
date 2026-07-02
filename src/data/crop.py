import numpy as np


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
