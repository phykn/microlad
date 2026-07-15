from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..misc import require_int
from .image import load_gray, load_labels
from .patch import augment, crop, resize
from .segment import segment_otsu


class PatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        image_paths: Sequence[str | Path],
        *,
        crop_size: int,
        image_size: int,
        num_phases: int,
        segment: bool = False,
        augment: bool = False,
    ) -> None:
        require_int("crop_size", crop_size)
        require_int("image_size", image_size)
        require_int("num_phases", num_phases)

        if crop_size <= 0:
            raise ValueError("crop_size must be positive.")

        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        if num_phases < 2:
            raise ValueError("num_phases must be at least 2.")

        if num_phases > int(np.iinfo(np.uint8).max) + 1:
            raise ValueError("num_phases must be at most 256 for uint8 images.")

        self.crop_size = crop_size
        self.image_size = image_size
        self.num_phases = num_phases
        self.augment = augment
        self.segment = segment
        self.paths = tuple(Path(path) for path in image_paths)

        if not self.paths:
            raise ValueError("image_paths must not be empty.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        labels = self._load_labels(path)
        patch = crop(labels, self.crop_size)
        patch = resize(patch, self.image_size)

        if self.augment:
            patch = augment(patch)

        labels = torch.from_numpy(np.array(patch, copy=True))
        fractions = (
            torch.bincount(
                labels.flatten().to(torch.long),
                minlength=self.num_phases,
            ).to(torch.float32)
            / labels.numel()
        )
        return labels.unsqueeze(0).to(torch.float32), fractions

    def _load_labels(self, path: Path) -> np.ndarray:
        if self.segment:
            return segment_otsu(load_gray(path), self.num_phases)

        labels = load_labels(path)
        if labels.max() >= self.num_phases:
            raise ValueError(
                f"phase image {path} must contain values from 0 "
                f"to {self.num_phases - 1}."
            )
        return labels
