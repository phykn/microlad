import argparse
from collections.abc import Mapping
import os
from pathlib import Path

from src.build import (
    build_dataset,
    build_loader,
    build_model,
    build_optimizer,
    build_trainer,
)
from src.misc import load_config, save_config
from src.train import distributed


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "model.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    config_path = Path(DEFAULT_CONFIG).resolve()
    args = argparse.Namespace(**load_config(config_path))
    data_dir = getattr(args, "data_dir", None)
    image_paths = getattr(args, "image_paths", None)
    if not isinstance(data_dir, Mapping):
        parser.error("data.data_dir must define axes 0, 1, and 2")
    if image_paths is not None:
        parser.error("data.image_paths is not supported")
    if set(data_dir) != {0, 1, 2}:
        parser.error("data.data_dir must contain exactly axes 0, 1, and 2")
    dirs = {}
    for axis, value in data_dir.items():
        path = Path(value)
        if not path.is_absolute():
            path = config_path.parent / path
        dirs[axis] = path.resolve()
    args.data_dir = dirs
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
