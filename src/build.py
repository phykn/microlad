import argparse
from collections.abc import Iterable, Iterator, Mapping
from typing import Any
import torch
from torch.utils.data import DataLoader, Sampler

from .data import AxisPatchDataset, load_axis_images
from .data.axes import AXES
from .diffusion import DDPMProcess, DiffusionLoss
from .misc import require_int
from .model import MPDDUNet
from .train import MPDDTrainer


_MODEL_KEYS = ("size", "num_phases", "base_ch", "time_dim")


class _AxisBalancedSampler(Sampler[int]):
    def __init__(self, dataset: AxisPatchDataset) -> None:
        self.condition_indices = tuple(
            dataset.condition_indices[condition]
            for condition in AXES
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
) -> AxisPatchDataset:
    paths = getattr(args, "image_paths", None)
    data_dir = getattr(args, "data_dir", None)
    if data_dir is None or paths is not None:
        raise ValueError(
            "training requires axis data_dir values and does not support image_paths."
        )
    paths, conditions = load_axis_images(data_dir)
    return AxisPatchDataset(
        paths,
        conditions,
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

    if not isinstance(dataset, AxisPatchDataset):
        raise TypeError("dataset must be an AxisPatchDataset.")
    sampler: Sampler[int] = _AxisBalancedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def build_model(
    cfg: argparse.Namespace | Mapping[str, Any],
    *,
    label: str = "model config",
) -> MPDDUNet:
    vals = vars(cfg) if isinstance(cfg, argparse.Namespace) else cfg
    miss = [key for key in _MODEL_KEYS if key not in vals]
    if miss:
        raise ValueError(f"{label} is missing required value: {', '.join(miss)}")
    return MPDDUNet(
        num_phases=vals["num_phases"],
        image_size=vals["size"],
        base_ch=vals["base_ch"],
        time_dim=vals["time_dim"],
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
        loss=DiffusionLoss(
            build_diffusion(args, device=device),
            anchor_loss_weight=getattr(args, "anchor_loss_weight", 0.0),
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
