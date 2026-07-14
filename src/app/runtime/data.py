import argparse
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import default_collate

from src.pipeline.data import PatchDataset


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_dataset(args: argparse.Namespace) -> PatchDataset:
    image_paths = getattr(args, "image_paths", None)
    if image_paths is None:
        root = Path(args.data_dir)
        image_paths = sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]

    return PatchDataset(
        image_paths,
        crop_size=args.crop_size,
        image_size=args.size,
        num_phases=args.num_phases,
        segment=args.segment,
        augment=args.augment,
    )


def build_loader(
    dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Iterator:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    while True:
        indices = torch.randint(0, len(dataset), (args.batch_size,)).tolist()
        batch = default_collate([dataset[index] for index in indices])
        if device.type == "cuda":
            if isinstance(batch, torch.Tensor):
                batch = batch.pin_memory()
            else:
                batch = [item.pin_memory() for item in batch]
        yield batch
