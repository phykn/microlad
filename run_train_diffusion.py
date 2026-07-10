import argparse
import os

from src.app.runtime import (
    build_dataset,
    build_denoiser,
    build_diffusion_trainer,
    build_loader,
    build_optimizer,
    cleanup_distributed,
    copy_vae_run,
    apply_vae_defaults,
    load_defaults,
    load_run_vae,
    save_run_config,
    setup_device,
    wrap_distributed,
)


DEFAULT_CONFIG = "config/diffusion.yaml"


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
    apply_vae_defaults(args)
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
        vae = load_run_vae(args.vae_run_dir, device)
        model = wrap_distributed(
            build_denoiser(args).to(device),
            local_rank=local_rank,
            distributed=distributed,
        )
        optimizer = build_optimizer(model, args)

        trainer = build_diffusion_trainer(
            model=model,
            vae=vae,
            loader=loader,
            optimizer=optimizer,
            args=args,
            device=device,
        )
        if rank == 0:
            copy_vae_run(args.vae_run_dir, trainer.run_dir)
            save_run_config(trainer.run_dir, args, name="diffusion")
            print(f"Training diffusion steps={args.steps} save_dir={trainer.run_dir}")
        trainer.train()
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
