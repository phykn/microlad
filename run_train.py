import argparse
import os
import sys
from collections.abc import Mapping
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
_COLORS = ("36", "35", "34", "33", "32", "31")


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    try:
        return load_train_config(DEFAULT_CONFIG)
    except ValueError as exc:
        parser.error(str(exc))


def _print_config(cfg: TrainConfig) -> None:
    vals = cfg.as_dict()
    keys = [key for items in vals.values() for key, _ in _flatten(items)]
    width = max(map(len, keys))
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    print(_paint("=== MPDD TRAINING CONFIG ===", "1;37", color))
    for idx, (section, items) in enumerate(vals.items()):
        code = _COLORS[idx % len(_COLORS)]
        print(_paint(f"[{section.upper()}]", f"1;{code}", color))
        for key, val in _flatten(items):
            label = _paint(f"  {key:<{width}}", code, color)
            print(f"{label}  {_format(val)}")
    print(_paint("============================", "1;37", color))


def _flatten(vals: Mapping[object, object], root: str = ""):
    for key, val in vals.items():
        name = f"{root}.{key}" if root else str(key)
        if isinstance(val, Mapping):
            yield from _flatten(val, name)
        else:
            yield name, val


def _format(val: object) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return str(val).lower()
    return str(val)


def _paint(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def main() -> None:
    cfg = parse_args()
    device, local_rank, enabled = distributed.setup()
    trainer = None
    try:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            _print_config(cfg)
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
