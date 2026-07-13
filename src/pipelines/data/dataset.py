from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.pipelines.data.images import load_gray_image, load_phase_image
from src.pipelines.data.segmentation import segment_otsu
from src.pipelines.data.transforms import augment_patch, crop_square, resize_patch
from src.validation import require_int


class PatchDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
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
        self.image_paths = [Path(image_path) for image_path in image_paths]

        if not self.image_paths:
            raise ValueError("image_paths must not be empty.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]

        if self.segment:
            image = segment_otsu(load_gray_image(path), self.num_phases)
        else:
            image = load_phase_image(path)
            if image.min() < 0 or image.max() >= self.num_phases:
                raise ValueError(
                    f"phase image {path} must contain values from 0 "
                    f"to {self.num_phases - 1}."
                )

        patch = crop_square(image, self.crop_size)
        patch = resize_patch(patch, self.image_size)

        if self.augment:
            patch = augment_patch(patch)

        return torch.from_numpy(patch.astype(np.float32, copy=True)).unsqueeze(0)
