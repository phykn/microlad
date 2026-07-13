from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AnchorSlice:
    image: np.ndarray
    axis: int
    index: int


@dataclass(frozen=True)
class VolumeAnchor:
    image: torch.Tensor
    axis: int
    index: int
    start: int = 0
