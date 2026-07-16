import argparse
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
    if hasattr(args, "axis_data"):
        parser.error("data.axis_data is no longer supported; use data.axis_manifest")
    axis_manifest = getattr(args, "axis_manifest", None)
    data_dir = getattr(args, "data_dir", None)
    image_paths = getattr(args, "image_paths", None)
    if axis_manifest is None and data_dir is None and image_paths is None:
        parser.error(
            "data.axis_manifest, data.data_dir, or data.image_paths is required "
            "in the config file"
        )
    if axis_manifest is not None and (data_dir is not None or image_paths is not None):
        parser.error(
            "data.axis_manifest is mutually exclusive with data.data_dir and "
            "data.image_paths"
        )
    if axis_manifest is not None:
        path = Path(axis_manifest)
        if not path.is_absolute():
            path = config_path.parent / path
        args.axis_manifest = path.resolve()
    num_axis_conditions = getattr(args, "num_axis_conditions", 0)
    if axis_manifest is not None and num_axis_conditions != 3:
        parser.error(
            "model.num_axis_conditions must be 3 when data.axis_manifest is configured"
        )
    if axis_manifest is None and num_axis_conditions != 0:
        parser.error(
            "data.axis_manifest is required when model.num_axis_conditions is nonzero"
        )
    anchor_conditioning = getattr(args, "anchor_conditioning", False)
    anchor_release_step = getattr(args, "anchor_release_step", 0)
    if not isinstance(anchor_conditioning, bool):
        parser.error("model.anchor_conditioning must be a boolean")
    if (
        not isinstance(anchor_release_step, int)
        or isinstance(anchor_release_step, bool)
        or anchor_release_step < 0
    ):
        parser.error("model.anchor_release_step must be a non-negative integer")
    if not anchor_conditioning and anchor_release_step != 0:
        parser.error(
            "model.anchor_release_step requires model.anchor_conditioning=true"
        )
    timesteps = getattr(args, "timesteps", None)
    if timesteps is not None and anchor_release_step >= timesteps:
        parser.error("model.anchor_release_step must be smaller than timesteps")
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
