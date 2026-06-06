import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from build import build_dataset, build_loader, build_trainer, build_unet, build_vae
from models import DDPM


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_device() -> tuple[torch.device, int, bool]:
    if not is_distributed():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, False

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank), rank, True


def cleanup_distributed(enabled: bool) -> None:
    if enabled:
        dist.destroy_process_group()


def parse_args_from_list(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    defaults = {}
    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--unet-ckpt", default=None)
    parser.add_argument("--output-dir", default="output/slice_conditioned")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--slice-index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--latent-ch", type=int, default=4)
    parser.add_argument("--base-ch", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--max-slices", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--val-freq", type=int, default=500)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("--data-dir is required unless provided by --config")
    if args.vae_ckpt is None:
        parser.error("--vae-ckpt is required unless provided by --config")
    return args


def parse_args() -> argparse.Namespace:
    return parse_args_from_list()


def main() -> None:
    args = parse_args()
    device, local_rank, distributed = setup_device()
    rank = int(os.environ.get("RANK", "0"))

    dataset = build_dataset(args)
    loader = build_loader(dataset, args, device=device, distributed=distributed)
    vae = build_vae(args, device)
    unet = build_unet(args, device)
    if distributed:
        unet = DistributedDataParallel(unet, device_ids=[local_rank])

    ddpm = DDPM(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.min_lr,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(
        unet=unet,
        vae=vae,
        ddpm=ddpm,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=output_dir,
        max_grad_norm=args.max_grad_norm,
        accum_steps=args.accum_steps,
        rank=rank,
        condition_dropout=args.condition_dropout,
    )

    if rank == 0:
        print(f"Training steps={args.steps} save_dir={output_dir}")
    trainer.train(steps=args.steps, val_freq=args.val_freq, save_freq=args.save_freq)

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
