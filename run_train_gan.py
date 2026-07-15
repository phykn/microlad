import argparse
import os

from src.app.runtime import (
    apply_gan_defaults,
    build_critic,
    build_dataset,
    build_gan_trainer,
    build_generator,
    build_loader,
    cleanup_distributed,
    copy_vae_run,
    load_defaults,
    load_run_vae,
    save_run_config,
    setup_device,
    wrap_distributed,
)
from src.pipeline.data import FakeLatentDataset


DEFAULT_CONFIG = "config/gan.yaml"


def parse_args_from_list(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    args = argparse.Namespace(**load_defaults(DEFAULT_CONFIG))
    if (
        getattr(args, "data_dir", None) is None
        and getattr(args, "image_paths", None) is None
    ):
        parser.error("data.data_dir or data.image_paths is required in the config file")
    if getattr(args, "vae_run_dir", None) is None:
        parser.error("output.vae_run_dir is required in the config file")
    if getattr(args, "fake_dir", None) is None:
        parser.error("data.fake_dir is required in the config file")
    apply_gan_defaults(args)
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
        fake_dataset = FakeLatentDataset(args.fake_dir)
        vae = load_run_vae(args.vae_run_dir, device)
        generator = wrap_distributed(
            build_generator(args).to(device),
            local_rank=local_rank,
            distributed=distributed,
        )
        critic = wrap_distributed(
            build_critic(args).to(device),
            local_rank=local_rank,
            distributed=distributed,
        )
        trainer = build_gan_trainer(
            generator,
            critic,
            vae,
            loader,
            fake_dataset,
            args,
            device,
        )
        if rank == 0:
            copy_vae_run(args.vae_run_dir, trainer.run_dir)
            save_run_config(trainer.run_dir, args, name="gan")
            print(f"Training GAN steps={args.steps} save_dir={trainer.run_dir}")
        trainer.train()
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
