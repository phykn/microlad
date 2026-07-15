import argparse
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import default_collate

from .data import PatchDataset
from .diffusion import DDPMProcess, DiffusionLoss
from .model import MPDDUNet
from .train import MPDDTrainer


_IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


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
    loader: Iterator,
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
