import numpy as np


def augment_patch(patch: np.ndarray) -> np.ndarray:
    transform = int(np.random.randint(0, 8))
    if transform >= 4:
        patch = np.fliplr(patch)
        transform -= 4
    return np.rot90(patch, transform)
