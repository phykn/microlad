import argparse
from collections.abc import Iterator
from pathlib import Path

import torch

from src.pipelines.data import PatchDataset
from src.modeling.diffusion import DDPMProcess, DiffusionLoss, TimeUNet
from src.pipelines.training import DiffusionTrainer, VAETrainer
from src.modeling.vae import PatchVAE, VAELoss

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_MISSING = object()

def _get_arg(args: argparse.Namespace, *names: str, default=_MISSING):
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)

    if default is not _MISSING:
        return default

    raise AttributeError(f"missing config value: {' or '.join(names)}")


def _make_vae(args: argparse.Namespace) -> PatchVAE:
    return PatchVAE(
        image_size=_get_arg(args, "vae_image_size", "image_size", "size"),
        latent_size=_get_arg(args, "vae_latent_size", "latent_size", default=16),
        latent_ch=_get_arg(args, "vae_latent_ch", "latent_ch"),
        num_phases=_get_arg(args, "vae_num_phases", "num_phases", default=3),
        base_ch=_get_arg(args, "vae_base_ch", "base_ch", default=64),
        max_ch=_get_arg(args, "vae_max_ch", "max_ch", default=512),
    )


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
        size=args.size,
        num_phases=args.num_phases,
        segment=getattr(args, "segment", False),
        augment=getattr(args, "augment", False),
    )


def build_loader(
    dataset: PatchDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Iterator:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    while True:
        indices = torch.randint(0, len(dataset), (args.batch_size,)).tolist()
        batch = torch.stack([dataset[index] for index in indices])
        yield batch.pin_memory() if device.type == "cuda" else batch


def build_vae(args: argparse.Namespace) -> PatchVAE:
    return _make_vae(args)


def build_denoiser(args: argparse.Namespace) -> TimeUNet:
    return TimeUNet(
        latent_ch=args.latent_ch,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
    )


def build_ddpm(
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
        weight_decay=getattr(args, "weight_decay", 0.0),
    )


def build_vae_trainer(
    model: torch.nn.Module,
    loader: Iterator,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> VAETrainer:
    loss_fn = VAELoss(
        beta=args.beta,
        num_phases=args.num_phases,
    )
    return VAETrainer(
        model=model,
        dataloader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        run_root=getattr(args, "run_root", "run"),
        save_every=args.save_every,
        clip_grad_norm=args.clip_grad_norm,
    )


def build_diffusion_trainer(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    loader: Iterator,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> DiffusionTrainer:
    return DiffusionTrainer(
        model=model,
        vae=vae,
        dataloader=loader,
        loss_fn=DiffusionLoss(build_ddpm(args, device=device)),
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        run_root=getattr(args, "run_root", "run"),
        save_every=args.save_every,
        clip_grad_norm=args.clip_grad_norm,
    )
