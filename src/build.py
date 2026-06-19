import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.dataset import PatchDataset
from src.loss import UNetDiffusionLoss, VAELoss
from src.models import DDPM, PatchVAE, TimeUNet
from src.trainer import Trainer


def _flatten_config(config: dict) -> dict:
    defaults = {}

    def visit(values: dict) -> None:
        for key, value in values.items():
            if isinstance(value, dict):
                visit(value)
                continue
            if key in defaults:
                raise ValueError(f"Duplicate config key: {key}")
            defaults[key] = value

    visit(config)
    return defaults


def load_config_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("config file must contain a mapping.")
    return _flatten_config(config)


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_device() -> tuple[torch.device, int, bool]:
    if not is_distributed():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl" if dist.is_nccl_available() else "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return device, local_rank, True


def cleanup_distributed(enabled: bool) -> None:
    if enabled:
        dist.destroy_process_group()


def wrap_distributed(
    model: torch.nn.Module, local_rank: int, distributed: bool
) -> torch.nn.Module:
    if not distributed:
        return model
    if next(model.parameters()).device.type != "cuda":
        return DistributedDataParallel(model)
    return DistributedDataParallel(model, device_ids=[local_rank])


def _load_model_checkpoint(
    target: torch.nn.Module, checkpoint, *state_dict_keys: str
) -> None:
    if isinstance(checkpoint, dict):
        for key in state_dict_keys:
            if key in checkpoint:
                target.load_state_dict(checkpoint[key])
                return
    target.load_state_dict(checkpoint)


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_dataset(args: argparse.Namespace) -> PatchDataset:
    return PatchDataset(
        args.data_dir,
        patch_size=args.patch_size,
    )


def build_loader(
    dataset, args: argparse.Namespace, device: torch.device, distributed: bool
):
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def load_frozen_vae(args: argparse.Namespace, device: torch.device) -> PatchVAE:
    vae = PatchVAE(latent_ch=args.latent_ch).to(device)
    _load_model_checkpoint(
        vae, torch.load(args.vae_ckpt, map_location=device), "vae", "model"
    )
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def load_unet(args: argparse.Namespace, device: torch.device) -> TimeUNet:
    unet = TimeUNet(
        latent_ch=args.latent_ch,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
    )
    if getattr(args, "unet_ckpt", None):
        _load_model_checkpoint(
            unet, torch.load(args.unet_ckpt, map_location="cpu"), "model"
        )
    return unet.to(device)


def build_optimizer(
    model: torch.nn.Module, args: argparse.Namespace
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=args.lr)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.min_lr,
    )


def build_vae_trainer(
    vae,
    loader,
    optimizer,
    scheduler,
    save_dir,
    kl_weight: float,
    ssim_weight: float,
    max_grad_norm: float,
    accum_steps: int,
    rank: int,
) -> Trainer:
    return Trainer(
        model=vae,
        train_loader=loader,
        criterion=VAELoss(kl_weight=kl_weight, ssim_weight=ssim_weight),
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        max_grad_norm=max_grad_norm,
        accum_steps=accum_steps,
        rank=rank,
    )


def build_unet_trainer(
    unet,
    vae,
    ddpm: DDPM,
    loader,
    optimizer,
    scheduler,
    save_dir,
    max_grad_norm: float,
    accum_steps: int,
    rank: int,
) -> Trainer:
    criterion = UNetDiffusionLoss(vae=vae, ddpm=ddpm)
    return Trainer(
        model=unet,
        train_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        max_grad_norm=max_grad_norm,
        accum_steps=accum_steps,
        rank=rank,
    )
