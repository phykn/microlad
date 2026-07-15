import argparse
import os

from src.build import (
    build_dataset,
    build_loader,
    build_model,
    build_optimizer,
    build_trainer,
)
from src.misc import load_config, save_config
from src.train import distributed


DEFAULT_CONFIG = "config/model.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    args = argparse.Namespace(**load_config(DEFAULT_CONFIG))
    if (
        getattr(args, "data_dir", None) is None
        and getattr(args, "image_paths", None) is None
    ):
        parser.error("data.data_dir or data.image_paths is required in the config file")
    return args


def main() -> None:
    args = parse_args()
    device, local_rank, enabled = distributed.setup()
    trainer = None
    try:
        rank = int(os.environ.get("RANK", "0"))
        dataset = build_dataset(args)
        loader = build_loader(dataset, args, device=device)
        model = distributed.wrap(
            build_model(args).to(device),
            local_rank=local_rank,
            enabled=enabled,
        )
        optimizer = build_optimizer(model, args)
        trainer = build_trainer(
            model=model,
            loader=loader,
            optimizer=optimizer,
            args=args,
            device=device,
        )
        if rank == 0:
            save_config(trainer.run_dir, args, name="model")
            print(f"Training MPDD steps={args.steps} save_dir={trainer.run_dir}")
        trainer.train()
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            distributed.cleanup(enabled)


if __name__ == "__main__":
    main()
