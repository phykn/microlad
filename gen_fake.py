import argparse
from pathlib import Path

import torch

from src.app.runtime import load_defaults, load_predictor
from src.pipeline.data import generate_lmpdd_fakes


DEFAULT_CONFIG = Path("config/gen_fake.yaml")


def load_config(path: str | Path) -> dict:
    values = load_defaults(path, label="fake config")
    required = {"vae_run_dir", "diffusion_run_dir", "num_volumes", "progress"}
    if set(values) != required:
        raise ValueError(f"fake config must contain: {', '.join(sorted(required))}")
    for name in ("vae_run_dir", "diffusion_run_dir"):
        value = values[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"fake config {name} must be a path.")
    value = values["num_volumes"]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("fake config num_volumes must be a positive integer.")
    if not isinstance(values["progress"], bool):
        raise ValueError("fake config progress must be a boolean.")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline L-MPDD latent volumes for critic fake data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration without loading models or generating data",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    values = load_config(args.config)
    if args.check:
        print("critic fake config is valid")
        return
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_predictor(
        values["vae_run_dir"],
        values["diffusion_run_dir"],
        gan_run_dir=None,
        device=device,
    )
    output_dir = Path("fake")
    paths = generate_lmpdd_fakes(
        predictor.sampler,
        predictor.vae,
        output_dir,
        num_volumes=values["num_volumes"],
        progress=values["progress"],
    )
    print(f"Generated {len(paths)} L-MPDD fake volumes at {output_dir}")


if __name__ == "__main__":
    main()
