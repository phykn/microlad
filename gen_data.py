import argparse
from pathlib import Path

from src.misc import load_config
from src.simul import save_data


DEFAULT_CONFIG = "config/simul.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    path = parser.parse_args(argv).config
    return argparse.Namespace(**load_config(path))


def main() -> None:
    volumes, slices = save_data(**vars(parse_args()))
    print(f"volumes={len(volumes)} dir={volumes[0].parent}")
    print(f"slices={len(slices)} dir={slices[0].parent}")


if __name__ == "__main__":
    main()
