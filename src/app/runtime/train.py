import argparse
from collections.abc import Iterator

import torch

from src.app.runtime.model import build_ddpm
from src.modeling.diffusion import DiffusionLoss
from src.modeling.vae import VAELoss
from src.pipelines.training import DiffusionTrainer, VAETrainer


def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
        phase_balance=args.phase_balance,
    )
    return VAETrainer(
        model=model,
        dataloader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        run_root=args.run_root,
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
        run_root=args.run_root,
        save_every=args.save_every,
        clip_grad_norm=args.clip_grad_norm,
        ema_decay=args.ema_decay,
    )
