import argparse
import os

from src.build import (
    build_dataset,
    build_loader,
    build_optimizer,
    build_scheduler,
    build_unet_trainer,
    cleanup_distributed,
    ensure_output_dir,
    load_config_defaults,
    load_frozen_vae,
    load_unet,
    setup_device,
    wrap_distributed,
)
from src.models import DDPM


DEFAULT_CONFIG = "config/train_unet.yaml"


def parse_args_from_list(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    args = argparse.Namespace(**load_config_defaults(DEFAULT_CONFIG))
    if getattr(args, "data_dir", None) is None:
        parser.error("data.data_dir is required in the config file")
    if getattr(args, "vae_ckpt", None) is None:
        parser.error("checkpoints.vae_ckpt is required in the config file")
    return args


def parse_args() -> argparse.Namespace:
    return parse_args_from_list()


def main() -> None:
    args = parse_args()
    device, local_rank, distributed = setup_device()
    try:
        rank = int(os.environ.get("RANK", "0"))

        dataset = build_dataset(args)
        loader = build_loader(dataset, args, device=device, distributed=distributed)
        vae = load_frozen_vae(args, device)
        unet = load_unet(args, device)
        unet = wrap_distributed(unet, local_rank=local_rank, distributed=distributed)

        ddpm = DDPM(timesteps=args.timesteps, device=device)
        optimizer = build_optimizer(unet, args)
        scheduler = build_scheduler(optimizer, args)

        output_dir = ensure_output_dir(args.output_dir)
        trainer = build_unet_trainer(
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
        )

        if rank == 0:
            print(f"Training steps={args.steps} save_dir={output_dir}")
        trainer.train(steps=args.steps, save_freq=args.save_freq)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
