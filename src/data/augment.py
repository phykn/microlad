import numpy as np


def augment_patch(patch: np.ndarray) -> np.ndarray:
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1]:
        raise ValueError("patch must be a square 2D array.")

    transform = int(np.random.randint(0, 8))

    if transform >= 4:
        patch = np.fliplr(patch)
        transform -= 4

    return np.rot90(patch, transform)
