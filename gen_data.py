import argparse
from pathlib import Path

from src.misc import load_mapping
from src.simul import save_simulation


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "simul.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    path = parser.parse_args(argv).config.resolve()
    config = load_mapping(path, label="simulation config")
    missing = {"output", "geometry"}.difference(config)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"simulation config is missing: {names}.")
    output = dict(config["output"])
    if "data_dir" in output:
        data_dir = Path(output["data_dir"])
        if not data_dir.is_absolute():
            data_dir = path.parent / data_dir
        output["data_dir"] = data_dir.resolve()
    config["output"] = output
    return argparse.Namespace(**config)


def main() -> None:
    args = parse_args()
    cfg = dict(args.output)
    try:
        data_dir = cfg.pop("data_dir")
        count = cfg.pop("count")
        axes = cfg.pop("axes")
    except KeyError as exc:
        raise ValueError(f"output is missing: {exc.args[0]}.") from exc
    if cfg:
        names = ", ".join(sorted(cfg))
        raise ValueError(f"unknown output settings: {names}.")

    vols, slices = save_simulation(
        data_dir,
        count=count,
        geometry=args.geometry,
        axes=axes,
    )
    print(f"volumes={len(vols)} dir={vols[0].parent}")
    print(
        "slices="
        f"{sum(len(paths) for paths in slices.values())} "
        f"dirs={','.join(str(paths[0].parent) for paths in slices.values())}"
    )


if __name__ == "__main__":
    main()
