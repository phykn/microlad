import argparse
import os

from src.build import (
    build_dataset,
    build_ddpm,
    build_loader,
    build_optimizer,
    build_scheduler,
    build_trainer,
    build_unet,
    build_vae,
    cleanup_distributed,
    ensure_output_dir,
    load_config_defaults,
    setup_device,
    wrap_distributed,
)


def parse_args_from_list(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    defaults = load_config_defaults(config_args.config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--vae-ckpt", default=None)
    parser.add_argument("--unet-ckpt", default=None)
    parser.add_argument("--output-dir", default="output/unet")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--latent-ch", type=int, default=4)
    parser.add_argument("--base-ch", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--val-freq", type=int, default=500)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1)
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
    unet = wrap_distributed(unet, local_rank=local_rank, distributed=distributed)

    ddpm = build_ddpm(args, device)
    optimizer = build_optimizer(unet, args)
    scheduler = build_scheduler(optimizer, args)

    output_dir = ensure_output_dir(args.output_dir)
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
    )

    if rank == 0:
        print(f"Training steps={args.steps} save_dir={output_dir}")
    trainer.train(steps=args.steps, val_freq=args.val_freq, save_freq=args.save_freq)

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
