from pathlib import Path

from PIL import Image
import numpy as np
import tifffile

from ..misc import require_int
from .sphere import make_volume


EDGE = 8
PALETTE = [0, 0, 0, 140, 140, 140, 255, 255, 255] + [0, 0, 0] * 253


def save_data(
    data_dir: str | Path,
    *,
    count: int,
    size: int,
    big_radius: int,
    big_fraction: float,
    small_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Save simulated volumes and their interior Z slices."""
    require_int("count", count)
    if count <= 0:
        raise ValueError("count must be positive.")

    gt_dir, train_dir = _make_dirs(data_dir, size)
    volumes = []
    slices = []
    for index in range(count):
        vol = make_volume(
            size=size,
            big_radius=big_radius,
            big_fraction=big_fraction,
            small_fraction=small_fraction,
            seed=seed + index,
        )
        stem = "volume" if count == 1 else f"volume_{index:03d}"
        slice_stem = "slice" if count == 1 else stem
        path, images = _save(vol, gt_dir, train_dir, stem, slice_stem)
        volumes.append(path)
        slices.extend(images)
    return volumes, slices


def _make_dirs(data_dir: str | Path, size: int) -> tuple[Path, Path]:
    if size <= 2 * EDGE:
        raise ValueError(f"size must exceed {2 * EDGE} to trim train slices.")

    root = Path(data_dir)
    dirs = root / "gt", root / "train"
    for path in dirs:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _save(
    vol: np.ndarray,
    gt_dir: Path,
    train_dir: Path,
    stem: str,
    slice_stem: str,
) -> tuple[Path, list[Path]]:
    volume_path = gt_dir / f"{stem}.tif"
    tifffile.imwrite(
        volume_path,
        vol,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "phase_labels": {
                "0": "background",
                "1": "small_sphere",
                "2": "big_sphere",
            },
        },
    )

    images = []
    for z in range(EDGE, vol.shape[0] - EDGE):
        path = train_dir / f"{slice_stem}_z_{z:03d}.png"
        image = Image.fromarray(vol[z])
        image.putpalette(PALETTE)
        image.save(path)
        images.append(path)
    return volume_path, images
