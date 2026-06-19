import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.helper import IMAGE_EXTENSIONS, load_image


class PatchDataset(Dataset):
    """Read SEM images and return 2D patches cropped in memory."""

    def __init__(
        self,
        root_dir: str | Path,
        patch_size: int = 64,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")

        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.paths = self._find_paths()

        if not self.paths:
            raise ValueError(
                f"No {patch_size}x{patch_size} patches found under {self.root_dir}."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self._load_slice(self.paths[index])
        patch = self._crop(image)
        return self._to_tensor(patch)

    def _load_slice(self, path: Path) -> np.ndarray:
        image = load_image(path)
        if image.ndim == 3:
            return image[random.randint(0, image.shape[0] - 1)]
        return image

    def _crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        x = random.randint(0, width - self.patch_size)
        y = random.randint(0, height - self.patch_size)
        return image[y : y + self.patch_size, x : x + self.patch_size]

    def _to_tensor(self, patch: np.ndarray) -> torch.Tensor:
        array = np.asarray(patch, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def _find_paths(self) -> list[Path]:
        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        paths: list[Path] = []
        for path in sorted(self.root_dir.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower().lstrip(".") not in IMAGE_EXTENSIONS
            ):
                continue
            image = load_image(path)
            if image.ndim not in (2, 3):
                continue
            height, width = image.shape[-2:]
            if height >= self.patch_size and width >= self.patch_size:
                paths.append(path)
        return paths
