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
from src.misc import save_config
from src.train import TrainConfig, distributed, load_train_config


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "model.yaml"


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    try:
        return load_train_config(DEFAULT_CONFIG)
    except ValueError as exc:
        parser.error(str(exc))


def main() -> None:
    cfg = parse_args()
    device, local_rank, enabled = distributed.setup()
    trainer = None
    try:
        rank = int(os.environ.get("RANK", "0"))
        dataset = build_dataset(cfg.data)
        loader = build_loader(dataset, cfg.data, device=device)
        model = distributed.wrap(
            build_model(cfg).to(device),
            local_rank=local_rank,
            enabled=enabled,
        )
        optimizer = build_optimizer(model, cfg.optimization)
        trainer = build_trainer(
            model=model,
            loader=loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
        )
        if rank == 0:
            save_config(trainer.run_dir, cfg.as_dict(), name="model")
            print(
                f"Training MPDD steps={cfg.training.steps} "
                f"save_dir={trainer.run_dir}"
            )
        trainer.train()
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            distributed.cleanup(enabled)


if __name__ == "__main__":
    main()
