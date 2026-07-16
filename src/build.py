import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from .data import AxisPatchDataset, PatchDataset, load_axis_manifest
from .data.manifest import AXIS_PLANES, IMAGE_EXTENSIONS
from .diffusion import DDPMProcess, DiffusionLoss
from .misc import require_int
from .model import MPDDUNet
from .model.factory import build_mpdd_model
from .train import MPDDTrainer


class _RandomSampler(Sampler[int]):
    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("dataset must not be empty.")
        self.size = size

    def __iter__(self) -> Iterator[int]:
        while True:
            yield int(torch.randint(self.size, ()).item())


class _AxisBalancedSampler(Sampler[int]):
    def __init__(self, dataset: AxisPatchDataset) -> None:
        self.condition_indices = tuple(
            dataset.condition_indices[condition]
            for condition in range(len(AXIS_PLANES))
        )
        if any(not indices for indices in self.condition_indices):
            raise ValueError(
                "axis dataset must contain images for conditions 0, 1, and 2."
            )

    def __iter__(self) -> Iterator[int]:
        while True:
            for condition in torch.randperm(len(self.condition_indices)).tolist():
                indices = self.condition_indices[condition]
                position = int(torch.randint(len(indices), ()).item())
                yield indices[position]


def build_dataset(
    args: argparse.Namespace,
) -> PatchDataset | AxisPatchDataset:
    if hasattr(args, "axis_data"):
        raise ValueError(
            "axis_data is no longer supported; use axis_manifest instead."
        )
    axis_manifest = getattr(args, "axis_manifest", None)
    paths = getattr(args, "image_paths", None)
    data_dir = getattr(args, "data_dir", None)

    if axis_manifest is not None:
        if data_dir is not None or paths is not None:
            raise ValueError(
                "axis_manifest is mutually exclusive with data_dir and image_paths."
            )
        if getattr(args, "axis_sampling", "balanced") != "balanced":
            raise ValueError("axis_sampling must be 'balanced'.")
        paths, conditions = load_axis_manifest(axis_manifest)
        return AxisPatchDataset(
            paths,
            conditions,
            crop_size=args.crop_size,
            image_size=args.size,
            num_phases=args.num_phases,
            segment=args.segment,
            augment=args.augment,
        )

    if paths is None:
        root = Path(args.data_dir)
        paths = sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
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

    if isinstance(dataset, AxisPatchDataset):
        if getattr(args, "axis_sampling", "balanced") != "balanced":
            raise ValueError("axis_sampling must be 'balanced'.")
        sampler: Sampler[int] = _AxisBalancedSampler(dataset)
    else:
        sampler = _RandomSampler(len(dataset))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def build_model(args: argparse.Namespace) -> MPDDUNet:
    return build_mpdd_model(vars(args))


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
        loss=DiffusionLoss(
            build_diffusion(args, device=device),
            anchor_loss_weight=getattr(args, "anchor_loss_weight", 0.0),
            anchor_phase_loss_weight=getattr(
                args,
                "anchor_phase_loss_weight",
                0.0,
            ),
        ),
        optimizer=optimizer,
        num_phases=args.num_phases,
        steps=args.steps,
        device=device,
        run_root=args.run_root,
        save_every=args.save_every,
        clip_grad_norm=args.clip_grad_norm,
        ema_decay=args.ema_decay,
        condition_dropout=getattr(args, "condition_dropout", 0.1),
        anchor_empty_probability=getattr(
            args,
            "anchor_empty_probability",
            0.2,
        ),
        warmup_steps=getattr(args, "warmup_steps", 0),
    )
