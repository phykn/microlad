from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.pipeline.data.images import load_gray_image, load_phase_image
from src.pipeline.data.segmentation import segment_otsu
from src.pipeline.data.transforms import augment_patch, crop_square, resize_patch
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

        image = torch.from_numpy(patch.astype(np.float32, copy=True)).unsqueeze(0)
        phase_fractions = torch.bincount(
            image.to(torch.long).flatten(),
            minlength=self.num_phases,
        ).to(torch.float32) / image.numel()
        return image, phase_fractions


class FakeLatentDataset(Dataset):
    def __init__(self, root: str | Path) -> None:
        self.paths = sorted(Path(root).glob("*.pt"))
        if not self.paths:
            raise ValueError(f"fake latent directory contains no .pt files: {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        latent = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
            raise ValueError("fake latent must have shape [C, D, H, W].")
        if not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError("fake latent must be a finite floating point tensor.")
        return latent


@torch.no_grad()
def generate_lmpdd_fakes(
    sampler,
    vae: torch.nn.Module,
    output_dir: str | Path,
    *,
    condition_dataset: Dataset,
    num_volumes: int,
    unconditional_ratio: float = 0.1,
    progress: bool = True,
) -> list[Path]:
    """Generate individual 3D L-MPDD latent files for critic training."""
    require_int("num_volumes", num_volumes)
    if num_volumes <= 0:
        raise ValueError("num_volumes must be positive.")
    if not isinstance(progress, bool):
        raise ValueError("progress must be a boolean.")
    if (
        not isinstance(unconditional_ratio, (int, float))
        or isinstance(unconditional_ratio, bool)
        or not 0.0 <= unconditional_ratio <= 1.0
    ):
        raise ValueError("unconditional_ratio must be between 0 and 1.")
    if len(condition_dataset) == 0:
        raise ValueError("condition_dataset must not be empty.")

    latent_ch = int(vae.latent_ch)
    latent_size = int(vae.latent_size)
    shape = (latent_size, latent_ch, latent_size, latent_size)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = [output / f"{index:05d}.pt" for index in range(num_volumes)]
    if existing := next((path for path in paths if path.exists()), None):
        raise FileExistsError(f"critic fake already exists: {existing}")

    unconditional_count = round(num_volumes * unconditional_ratio)
    if 0.0 < unconditional_ratio < 1.0 and num_volumes > 1:
        unconditional_count = min(max(unconditional_count, 1), num_volumes - 1)
    for index, path in enumerate(
        tqdm(paths, desc="critic fakes", disable=not progress)
    ):
        phase_fractions = None
        if index >= unconditional_count:
            sample_index = int(torch.randint(len(condition_dataset), (1,)).item())
            _, phase_fractions = condition_dataset[sample_index]
        latent = sampler.sample_lmpdd(
            shape,
            progress=False,
            phase_fractions=phase_fractions,
        )
        if (
            latent.shape != shape
            or not latent.is_floating_point()
            or not torch.isfinite(latent).all()
        ):
            raise ValueError(f"L-MPDD sampler must return floating tensor {shape}.")
        torch.save(latent.permute(1, 0, 2, 3).contiguous().cpu(), path)
    return paths
