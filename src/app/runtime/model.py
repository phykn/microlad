import argparse

import torch

from src.modeling.latent_gan import LatentCritic, LatentGenerator
from src.modeling.diffusion import DDPMProcess, TimeUNet
from src.modeling.vae import PatchVAE


def build_vae(args: argparse.Namespace) -> PatchVAE:
    return PatchVAE(
        image_size=args.size,
        latent_size=args.latent_size,
        latent_ch=args.latent_ch,
        num_phases=args.num_phases,
        phase_embedding_dim=getattr(args, "phase_embedding_dim", 4),
        base_ch=args.base_ch,
        max_ch=args.max_ch,
    )


def build_denoiser(args: argparse.Namespace) -> TimeUNet:
    return TimeUNet(
        latent_ch=args.latent_ch,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
        num_phases=args.num_phases,
    )


def build_generator(args: argparse.Namespace) -> LatentGenerator:
    return LatentGenerator(
        latent_ch=args.latent_ch,
        latent_size=args.latent_size,
        noise_ch=args.noise_ch,
        base_ch=args.generator_ch,
    )


def build_critic(args: argparse.Namespace) -> LatentCritic:
    return LatentCritic(
        latent_ch=args.latent_ch,
        base_ch=args.critic_ch,
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
