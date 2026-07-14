import argparse
from collections.abc import Iterator

import torch

from src.app.runtime.model import build_ddpm
from src.modeling.diffusion import DiffusionLoss
from src.modeling.vae import VAELoss
from src.pipeline.train import DiffusionTrainer, GANTrainer, VAETrainer


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
        condition_dropout=getattr(args, "condition_dropout", 0.1),
    )


def build_gan_trainer(
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    vae: torch.nn.Module,
    loader: Iterator,
    fake_loader: Iterator,
    args: argparse.Namespace,
    device: torch.device,
) -> GANTrainer:
    betas = tuple(args.betas)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.generator_lr,
        betas=betas,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=args.critic_lr,
        betas=betas,
    )
    return GANTrainer(
        generator=generator,
        critic=critic,
        vae=vae,
        dataloader=loader,
        fake_dataloader=fake_loader,
        generator_optimizer=generator_optimizer,
        critic_optimizer=critic_optimizer,
        steps=args.steps,
        critic_steps=args.critic_steps,
        gp_weight=args.gp_weight,
        clip_grad_norm=args.clip_grad_norm,
        save_every=args.save_every,
        device=device,
        run_root=args.run_root,
    )
