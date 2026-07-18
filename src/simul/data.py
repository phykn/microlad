from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from ..data.axes import AXES
from ..misc import require_int
from .geometry import make_geometry


PALETTE = [0, 0, 0, 140, 140, 140, 255, 255, 255] + [0, 0, 0] * 253


def save_simulation(
    data_dir: str | Path,
    *,
    count: int,
    geometry: Mapping[str, object],
    axes: Sequence[int] = AXES,
) -> tuple[list[Path], dict[int, list[Path]]]:
    require_int("count", count)
    if count <= 0:
        raise ValueError("count must be positive.")
    if not isinstance(geometry, Mapping):
        raise ValueError("geometry must be a mapping.")

    cfg = dict(geometry)
    if "size" not in cfg:
        raise ValueError("geometry must define size.")
    size = cfg["size"]
    require_int("size", size)

    axes = _validate_axes(axes)

    root = Path(data_dir)
    gt, dirs = _make_simulation_dirs(root, axes)
    vols: list[Path] = []
    slices = {axis: [] for axis in axes}

    for i in range(count):
        geo = make_geometry(**cfg)
        vol = geo.labels

        _validate_volume(vol, size)
        stem = "volume" if count == 1 else f"volume_{i:03d}"
        slice_name = "slice" if count == 1 else stem
        meta = {
            "generator": "packing",
            **geo.report.as_dict(),
        }
        path = gt / f"{stem}.tif"
        _write_volume(
            path,
            vol,
            meta=meta,
            labels=_phase_labels(float(cfg.get("big_elongation", 1.0))),
        )
        vols.append(path)

        for axis in axes:
            paths = _save_axis_slices(
                vol,
                dirs[axis],
                axis=axis,
                stem=slice_name,
            )
            slices[axis].extend(paths)

    return vols, slices


def _validate_axes(val: object) -> tuple[int, ...]:
    if not isinstance(val, Sequence) or isinstance(val, (str, bytes)):
        raise ValueError("axes must be a list of axis indices.")
    axes = tuple(val)
    if any(not isinstance(axis, int) or isinstance(axis, bool) for axis in axes):
        raise ValueError("axes must contain integer axis indices.")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must not contain duplicates.")
    if set(axes) != set(AXES):
        raise ValueError("axes must contain exactly 0, 1, and 2.")
    return axes


def _validate_volume(vol: np.ndarray, size: int) -> None:
    if vol.shape != (size, size, size):
        raise ValueError("generated volume must have shape [size, size, size].")
    if vol.dtype != np.uint8:
        raise ValueError("generated volume must have dtype uint8.")
    if not np.isin(vol, (0, 1, 2)).all():
        raise ValueError("generated volume may contain only labels 0, 1, and 2.")


def _make_simulation_dirs(
    root: Path,
    axes: tuple[int, ...],
) -> tuple[Path, dict[int, Path]]:
    gt = root / "gt"
    train = root / "train"
    for path in (gt, train):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    dirs = {axis: train / str(axis) for axis in axes}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return gt, dirs


def _save_axis_slices(
    vol: np.ndarray,
    dst: Path,
    *,
    axis: int,
    stem: str,
) -> list[Path]:
    stack = np.moveaxis(vol, axis, 0)
    paths = []
    for i in range(stack.shape[0]):
        path = dst / f"{stem}_{axis}_{i:03d}.png"
        _write_label_image(path, stack[i])
        paths.append(path)
    return paths


def _phase_labels(big_elongation: float) -> dict[str, str]:
    return {
        "0": "background",
        "1": "small_sphere",
        "2": "big_sphere" if big_elongation == 1.0 else "big_ellipse",
    }


def _write_volume(
    path: Path,
    vol: np.ndarray,
    *,
    meta: Mapping[str, object],
    labels: Mapping[str, str],
) -> None:
    tifffile.imwrite(
        path,
        vol,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "phase_labels": dict(labels),
            "simulation": dict(meta),
        },
    )


def _write_label_image(path: Path, data: np.ndarray) -> None:
    img = Image.fromarray(data)
    img.putpalette(PALETTE)
    img.save(path)
