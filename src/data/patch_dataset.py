from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augment import augment_patch
from src.data.crop import crop_square
from src.data.resize import resize_patch
from src.io import load_image
from src.segment import segment_multi_otsu


class PatchDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
        *,
        crop_size: int,
        size: int,
        num_phases: int,
        segment: bool = False,
        augment: bool = False,
    ) -> None:
        if crop_size <= 0:
            raise ValueError("crop_size must be positive.")
        if size <= 0:
            raise ValueError("size must be positive.")
        if num_phases < 2:
            raise ValueError("num_phases must be at least 2.")

        self.crop_size = crop_size
        self.size = size
        self.num_phases = num_phases
        self.augment = augment
        self.segment = segment
        self.image_paths = [Path(image_path) for image_path in image_paths]
        if not self.image_paths:
            raise ValueError("image_paths must not be empty.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        image = self._load_image(path)
        patch = crop_square(image, self.crop_size)
        patch = resize_patch(patch, self.size)
        if self.augment:
            patch = augment_patch(patch)
        patch = self._scale_phase(patch)
        return torch.from_numpy(patch.copy()).unsqueeze(0).float()

    def _load_image(self, path: Path) -> np.ndarray:
        image = load_image(path)
        if self.segment:
            return segment_multi_otsu(image, self.num_phases)
        self._validate_phase_image(image, path)
        return image

    def _validate_phase_image(self, image: np.ndarray, path: Path) -> None:
        if image.min() < 0 or image.max() >= self.num_phases:
            raise ValueError(
                f"phase image {path} must contain values from 0 to {self.num_phases - 1}."
            )

    def _scale_phase(self, patch: np.ndarray) -> np.ndarray:
        return patch.astype(np.float32) / (self.num_phases - 1) * 2.0 - 1.0
