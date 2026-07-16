import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from ..data.manifest import (
    AXIS_PLANES,
    PLANE_AXES,
    SCHEMA_VERSION,
    VOLUME_AXES,
)
from ..misc import require_int
from .dry import make_dry_geometry
from .sphere import make_geometry, make_volume


EDGE = 8
PALETTE = [0, 0, 0, 140, 140, 140, 255, 255, 255] + [0, 0, 0] * 253
DRY_PHASE_LABELS = {
    "0": "background",
    "1": "small_particle",
    "2": "big_particle",
}
SPHERE_PHASE_LABELS = {
    "0": "background",
    "1": "small_sphere",
    "2": "big_sphere",
}


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
    """Save legacy sphere volumes and their interior Z slices."""

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


def save_simulation(
    data_dir: str | Path,
    *,
    count: int,
    geometry: Mapping[str, object],
    export: Mapping[str, object] | None = None,
) -> tuple[list[Path], dict[str, list[Path]]]:
    """Generate configured volumes, orthogonal slices, and a dataset manifest."""

    require_int("count", count)
    if count <= 0:
        raise ValueError("count must be positive.")
    if not isinstance(geometry, Mapping):
        raise ValueError("geometry must be a mapping.")
    if export is not None and not isinstance(export, Mapping):
        raise ValueError("export must be a mapping.")

    settings = dict(geometry)
    mode = settings.pop("mode", "dry")
    if mode not in {"dry", "sphere"}:
        raise ValueError("geometry mode must be 'dry' or 'sphere'.")
    if "size" not in settings:
        raise ValueError("geometry must define size.")
    size = settings["size"]
    require_int("size", size)

    export_settings = {} if export is None else dict(export)
    planes = _validate_planes(export_settings.pop("planes", tuple(PLANE_AXES)))
    trim = export_settings.pop("trim", 0)
    require_int("trim", trim)
    if trim < 0 or size <= 2 * trim:
        raise ValueError("trim must be non-negative and leave at least one slice.")
    if export_settings:
        names = ", ".join(sorted(export_settings))
        raise ValueError(f"unknown export settings: {names}.")

    root = Path(data_dir)
    gt_dir, plane_dirs = _make_simulation_dirs(root, planes)
    volume_paths: list[Path] = []
    plane_paths = {plane: [] for plane in planes}
    records = []
    base_seed = settings.get("seed") if mode == "sphere" else None

    for index in range(count):
        volume_settings = dict(settings)
        if mode == "sphere" and base_seed is not None:
            require_int("seed", base_seed)
            volume_settings["seed"] = base_seed + index

        if mode == "dry":
            generated = make_dry_geometry(**volume_settings)
            volume = generated.labels
            report = generated.report.as_dict()
        else:
            volume, spheres = make_geometry(**volume_settings)
            fractions = _phase_fractions(volume)
            report = {
                "requested_fractions": [
                    1.0
                    - float(volume_settings["small_fraction"])
                    - float(volume_settings["big_fraction"]),
                    float(volume_settings["small_fraction"]),
                    float(volume_settings["big_fraction"]),
                ],
                "achieved_fractions": fractions,
                "particle_counts": {
                    "small": sum(label == 1 for _, _, label in spheres),
                    "big": sum(label == 2 for _, _, label in spheres),
                },
            }

        _validate_volume(volume, size)
        stem = "volume" if count == 1 else f"volume_{index:03d}"
        slice_stem = "slice" if count == 1 else stem
        metadata = {
            "generator": mode,
            **report,
        }
        if mode == "sphere":
            metadata["seed"] = volume_settings.get("seed")
        volume_path = gt_dir / f"{stem}.tif"
        _write_volume(
            volume_path,
            volume,
            metadata=metadata,
            phase_labels=(DRY_PHASE_LABELS if mode == "dry" else SPHERE_PHASE_LABELS),
        )
        volume_paths.append(volume_path)

        for plane in planes:
            paths = _save_plane_slices(
                volume,
                plane_dirs[plane],
                plane=plane,
                stem=slice_stem,
                trim=trim,
            )
            plane_paths[plane].extend(paths)

        records.append(
            {
                "name": stem,
                "path": volume_path.relative_to(root).as_posix(),
                **metadata,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "volume_axes": VOLUME_AXES,
        "axis_sources": {
            plane: plane_dirs[plane].relative_to(root).as_posix()
            for plane in AXIS_PLANES
        },
        "geometry": {"mode": mode, **settings},
        "export": {"planes": list(planes), "trim": trim},
        "volumes": records,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return volume_paths, plane_paths


def _validate_planes(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("planes must be a list of plane names.")
    planes = tuple(value)
    if not planes:
        raise ValueError("planes must not be empty.")
    if len(set(planes)) != len(planes):
        raise ValueError("planes must not contain duplicates.")
    unknown = set(planes).difference(PLANE_AXES)
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"unknown planes: {names}.")
    missing = set(AXIS_PLANES).difference(planes)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"planes must include {names}.")
    return planes


def _validate_volume(volume: np.ndarray, size: int) -> None:
    if volume.shape != (size, size, size):
        raise ValueError("generated volume must have shape [size, size, size].")
    if volume.dtype != np.uint8:
        raise ValueError("generated volume must have dtype uint8.")
    if not np.isin(volume, (0, 1, 2)).all():
        raise ValueError("generated volume may contain only labels 0, 1, and 2.")


def _make_simulation_dirs(
    root: Path,
    planes: tuple[str, ...],
) -> tuple[Path, dict[str, Path]]:
    gt_dir = root / "gt"
    train_dir = root / "train"
    for path in (gt_dir, train_dir):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    plane_dirs = {plane: train_dir / plane for plane in planes}
    for path in plane_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return gt_dir, plane_dirs


def _save_plane_slices(
    volume: np.ndarray,
    directory: Path,
    *,
    plane: str,
    stem: str,
    trim: int,
) -> list[Path]:
    axis, normal = PLANE_AXES[plane]
    stack = np.moveaxis(volume, axis, 0)
    paths = []
    for index in range(trim, stack.shape[0] - trim):
        path = directory / f"{stem}_{normal}_{index:03d}.png"
        _write_label_image(path, stack[index])
        paths.append(path)
    return paths


def _phase_fractions(volume: np.ndarray) -> list[float]:
    counts = np.bincount(volume.ravel(), minlength=3)
    return [float(value / volume.size) for value in counts]


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
    _write_volume(
        volume_path,
        vol,
        metadata={"generator": "sphere"},
        phase_labels=SPHERE_PHASE_LABELS,
    )

    images = []
    for z in range(EDGE, vol.shape[0] - EDGE):
        path = train_dir / f"{slice_stem}_z_{z:03d}.png"
        _write_label_image(path, vol[z])
        images.append(path)
    return volume_path, images


def _write_volume(
    path: Path,
    volume: np.ndarray,
    *,
    metadata: Mapping[str, object],
    phase_labels: Mapping[str, str],
) -> None:
    tifffile.imwrite(
        path,
        volume,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "phase_labels": dict(phase_labels),
            "simulation": dict(metadata),
        },
    )


def _write_label_image(path: Path, labels: np.ndarray) -> None:
    image = Image.fromarray(labels)
    image.putpalette(PALETTE)
    image.save(path)
