import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")


class PatchDataset(Dataset):
    """Read SEM images and return 2D patches cropped in memory."""

    def __init__(
        self,
        root_dir: str | Path,
        patch_size: int = 64,
        seed: int | None = None,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")

        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.rng = random.Random(seed)
        self.paths = self._load_paths()

        if not self.paths:
            raise ValueError(f"No {patch_size}x{patch_size} patches found under {self.root_dir}.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("L")
            x = self.rng.randint(0, image.width - self.patch_size)
            y = self.rng.randint(0, image.height - self.patch_size)
            patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))

        array = np.asarray(patch, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def _load_paths(self) -> list[Path]:
        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        paths: list[Path] = []
        for path in sorted(self.root_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower().lstrip(".") not in IMAGE_EXTENSIONS:
                continue
            with Image.open(path) as image:
                if image.width >= self.patch_size and image.height >= self.patch_size:
                    paths.append(path)
        return paths


AXIS_TO_INDEX = {"z": 0, "y": 1, "x": 2}


class SliceConditionDataset(PatchDataset):
    """Return a full 2D condition slice and its 3D slice position."""

    def __init__(
        self,
        root_dir: str | Path,
        patch_size: int = 64,
        axis: str = "z",
        slice_index: int = 0,
        num_conditions: int = 1,
        condition_axes: list[str] | None = None,
        condition_slice_indices: list[int] | None = None,
        seed: int | None = None,
    ) -> None:
        if axis not in AXIS_TO_INDEX:
            raise ValueError("axis must be one of: x, y, z.")
        if slice_index < 0:
            raise ValueError("slice_index must be non-negative.")
        if num_conditions <= 0:
            raise ValueError("num_conditions must be positive.")
        self.axis = AXIS_TO_INDEX[axis]
        self.slice_index = slice_index
        self.num_conditions = num_conditions
        self.condition_axes = self._normalize_axes(condition_axes, axis, num_conditions)
        self.condition_slice_indices = self._normalize_slice_indices(
            condition_slice_indices,
            slice_index,
            num_conditions,
        )
        super().__init__(root_dir=root_dir, patch_size=patch_size, seed=seed)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target = super().__getitem__(index)
        sample = {
            "target": target,
            "condition": target.clone(),
            "axis": self.axis,
            "slice_index": self.slice_index,
        }
        if self.num_conditions > 1:
            sample["conditions"] = torch.stack([target.clone() for _ in range(self.num_conditions)], dim=0)
            sample["axes"] = torch.tensor(self.condition_axes, dtype=torch.long)
            sample["slice_indices"] = torch.tensor(self.condition_slice_indices, dtype=torch.long)
        return sample

    @staticmethod
    def _normalize_axes(condition_axes: list[str] | None, axis: str, count: int) -> list[int]:
        axes = condition_axes if condition_axes is not None else [axis] * count
        if len(axes) != count:
            raise ValueError("condition_axes length must match num_conditions.")
        invalid = [item for item in axes if item not in AXIS_TO_INDEX]
        if invalid:
            raise ValueError("condition_axes must contain only x, y, or z.")
        return [AXIS_TO_INDEX[item] for item in axes]

    @staticmethod
    def _normalize_slice_indices(indices: list[int] | None, default: int, count: int) -> list[int]:
        values = indices if indices is not None else [default] * count
        if len(values) != count:
            raise ValueError("condition_slice_indices length must match num_conditions.")
        if any(item < 0 for item in values):
            raise ValueError("condition_slice_indices must be non-negative.")
        return [int(item) for item in values]
