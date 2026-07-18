from collections.abc import Mapping
from pathlib import Path


AXES = (0, 1, 2)
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_axis_images(
    data_dir: Mapping[int, str | Path],
) -> tuple[tuple[Path, ...], tuple[int, ...]]:
    if not isinstance(data_dir, Mapping):
        raise TypeError("data_dir must map axes 0, 1, and 2 to directories.")
    keys = set(data_dir)
    if keys != set(AXES):
        raise ValueError("data_dir must contain exactly axes 0, 1, and 2.")
    paths: list[Path] = []
    conditions: list[int] = []
    for axis in AXES:
        directory = Path(data_dir[axis])
        if not directory.is_dir():
            raise FileNotFoundError(
                f"axis {axis} data directory is required: {directory}"
            )
        images = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(
                f"axis {axis} data directory contains no supported images: {directory}"
            )
        paths.extend(images)
        conditions.extend([axis] * len(images))
    return tuple(paths), tuple(conditions)
