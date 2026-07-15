import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from .data import PatchDataset
from .diffusion import DDPMProcess, DiffusionLoss
from .misc import require_int
from .model import MPDDUNet
from .train import MPDDTrainer


_IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class _RandomSampler(Sampler[int]):
    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("dataset must not be empty.")
        self.size = size

    def __iter__(self) -> Iterator[int]:
        while True:
            yield int(torch.randint(self.size, ()).item())


def build_dataset(args: argparse.Namespace) -> PatchDataset:
    paths = getattr(args, "image_paths", None)
    if paths is None:
        root = Path(args.data_dir)
        paths = sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
    elif isinstance(paths, (str, Path)):
        paths = [paths]

    return PatchDataset(
        paths,
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
) -> DataLoader:
    workers = getattr(args, "num_workers", 0)
    require_int("batch_size", args.batch_size)
    require_int("num_workers", workers)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if workers < 0:
        raise ValueError("num_workers must be non-negative.")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=_RandomSampler(len(dataset)),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def build_model(args: argparse.Namespace) -> MPDDUNet:
    return MPDDUNet(
        num_phases=args.num_phases,
        image_size=args.size,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
    )


def build_diffusion(
    args: argparse.Namespace,
    device: torch.device,
) -> DDPMProcess:
    return DDPMProcess(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )


def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_trainer(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> MPDDTrainer:
    return MPDDTrainer(
        model=model,
        loader=loader,
        loss=DiffusionLoss(build_diffusion(args, device=device)),
        optimizer=optimizer,
        num_phases=args.num_phases,
        steps=args.steps,
        device=device,
        run_root=args.run_root,
        save_every=args.save_every,
        clip_grad_norm=args.clip_grad_norm,
        ema_decay=args.ema_decay,
        condition_dropout=getattr(args, "condition_dropout", 0.1),
        warmup_steps=getattr(args, "warmup_steps", 0),
    )
