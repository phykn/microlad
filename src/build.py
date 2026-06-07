import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import SliceConditionDataset
from models import CustomVAE, DDPM, SliceConditionedTimeUNet, TimeUNet
from trainer import Trainer
from loss import SliceConditionedDiffusionLoss


def build_dataset(args: argparse.Namespace) -> SliceConditionDataset:
    return SliceConditionDataset(
        args.data_dir,
        patch_size=args.patch_size,
        axis=args.axis,
        slice_index=args.slice_index,
        num_conditions=args.num_conditions,
        condition_axes=args.condition_axes,
        condition_slice_indices=args.condition_slice_indices,
    )


def build_loader(dataset, args: argparse.Namespace, device: torch.device, distributed: bool):
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def build_vae(args: argparse.Namespace, device: torch.device) -> CustomVAE:
    vae = CustomVAE(latent_ch=args.latent_ch).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["vae"])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def build_unet(args: argparse.Namespace, device: torch.device) -> SliceConditionedTimeUNet:
    unet = SliceConditionedTimeUNet(
        latent_ch=args.latent_ch,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
        max_slices=args.max_slices,
    )
    if getattr(args, "unet_ckpt", None):
        load_unet_checkpoint(unet, torch.load(args.unet_ckpt, map_location="cpu"))
    return unet.to(device)


def build_unet_checkpoint_source(latent_ch: int, base_ch: int, time_dim: int) -> TimeUNet:
    return TimeUNet(latent_ch=latent_ch, base_ch=base_ch, time_dim=time_dim)


def import_base_unet_weights(target: SliceConditionedTimeUNet, state_dict: dict[str, torch.Tensor]) -> None:
    target_state = target.state_dict()
    updates = {}
    for key, value in state_dict.items():
        target_key = f"unet.{key}"
        if target_key in target_state and target_state[target_key].shape == value.shape:
            updates[target_key] = value
    target_state.update(updates)
    target.load_state_dict(target_state)


def load_unet_checkpoint(target: SliceConditionedTimeUNet, checkpoint) -> None:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        target.load_state_dict(checkpoint["model"])
        return
    import_base_unet_weights(target, checkpoint)


def build_trainer(
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
    condition_dropout: float = 0.0,
) -> Trainer:
    criterion = SliceConditionedDiffusionLoss(
        vae=vae,
        ddpm=ddpm,
        condition_dropout=condition_dropout,
    )
    return Trainer(
        model=unet,
        train_loader=loader,
        valid_loader=None,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        max_grad_norm=max_grad_norm,
        accum_steps=accum_steps,
        rank=rank,
    )
