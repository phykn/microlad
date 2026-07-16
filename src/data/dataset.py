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


class AxisPatchDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Attach a categorical plane condition to each patch image."""

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        axis_conditions: Sequence[int],
        *,
        crop_size: int,
        image_size: int,
        num_phases: int,
        segment: bool = False,
        augment: bool = False,
    ) -> None:
        self.patch_dataset = PatchDataset(
            image_paths,
            crop_size=crop_size,
            image_size=image_size,
            num_phases=num_phases,
            segment=segment,
            augment=augment,
        )
        self.axis_conditions = tuple(axis_conditions)

        if len(self.axis_conditions) != len(self.patch_dataset):
            raise ValueError(
                "axis_conditions must contain one condition per image path."
            )

        condition_indices: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for index, condition in enumerate(self.axis_conditions):
            require_int("axis condition", condition)
            if condition not in condition_indices:
                raise ValueError("axis condition must be one of 0, 1, or 2.")
            condition_indices[condition].append(index)

        self.condition_indices = {
            condition: tuple(indices)
            for condition, indices in condition_indices.items()
        }

    @property
    def paths(self) -> tuple[Path, ...]:
        return self.patch_dataset.paths

    def __len__(self) -> int:
        return len(self.patch_dataset)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, fractions = self.patch_dataset[index]
        condition = torch.tensor(
            self.axis_conditions[index],
            dtype=torch.long,
        )
        return image, fractions, condition
