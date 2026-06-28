import argparse
import os

from src.build import (
    build_dataset,
    build_loader,
    build_vae,
    build_optimizer,
    build_vae_trainer,
    cleanup_distributed,
    load_config_defaults,
    save_run_config,
    setup_device,
    wrap_distributed,
)


DEFAULT_CONFIG = "config/vae.yaml"


def parse_args_from_list(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    args = argparse.Namespace(**load_config_defaults(DEFAULT_CONFIG))
    if (
        getattr(args, "data_dir", None) is None
        and getattr(args, "image_paths", None) is None
    ):
        parser.error("data.data_dir or data.image_paths is required in the config file")
    return args


def parse_args() -> argparse.Namespace:
    return parse_args_from_list()


def main() -> None:
    args = parse_args()
    device, local_rank, distributed = setup_device()
    trainer = None
    try:
        rank = int(os.environ.get("RANK", "0"))
        dataset = build_dataset(args)
        loader = build_loader(dataset, args, device=device)
        vae = wrap_distributed(
            build_vae(args).to(device),
            local_rank=local_rank,
            distributed=distributed,
        )
        optimizer = build_optimizer(vae, args)

        trainer = build_vae_trainer(
            model=vae,
            loader=loader,
            optimizer=optimizer,
            args=args,
            device=device,
        )
        if rank == 0:
            save_run_config(trainer.run_dir, args, name="vae")
            print(f"Training VAE steps={args.steps} save_dir={trainer.run_dir}")
        trainer.train()
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
