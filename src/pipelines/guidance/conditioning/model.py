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
    """Categorical slice constraint positioned inside a 3D output volume."""

    image: torch.Tensor
    axis: int
    index: int
    start: int = 0
