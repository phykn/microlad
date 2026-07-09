from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnchorSlice:
    image: np.ndarray
    axis: int
    index: int
