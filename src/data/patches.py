from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")


@dataclass(frozen=True)
class PatchSample:
    source_path: Path
    xy: tuple[int, int]


class PatchDataset(Dataset):
    """Read SEM images and return 2D patches cropped in memory."""

    def __init__(
        self,
        root_dir: str | Path,
        patch_size: int = 64,
        stride: int = 64,
        extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.extensions = tuple(ext.lower().lstrip(".") for ext in extensions)
        self.image_paths = self._find_images()
        self.samples = self._index_patches()

        if not self.samples:
            raise ValueError(f"No {patch_size}x{patch_size} patches found under {self.root_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        x, y = sample.xy
        with Image.open(sample.source_path) as image:
            image = image.convert("L")
            patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))

        array = np.asarray(patch, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return {
            "image": tensor,
            "source_path": str(sample.source_path),
            "xy": sample.xy,
        }

    def _find_images(self) -> list[Path]:
        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        paths: list[Path] = []
        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.suffix.lower().lstrip(".") in self.extensions:
                paths.append(path)
        return sorted(paths)

    def _index_patches(self) -> list[PatchSample]:
        samples: list[PatchSample] = []
        for path in self.image_paths:
            with Image.open(path) as image:
                width, height = image.size

            max_x = width - self.patch_size
            max_y = height - self.patch_size
            if max_x < 0 or max_y < 0:
                continue

            for y in range(0, max_y + 1, self.stride):
                for x in range(0, max_x + 1, self.stride):
                    samples.append(PatchSample(source_path=path, xy=(x, y)))

        return samples
