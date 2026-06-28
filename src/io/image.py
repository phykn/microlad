from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    with Image.open(Path(path)) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)
