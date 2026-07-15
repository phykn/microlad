import argparse
import math
from numbers import Real
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


PALETTE = [
    0,
    0,
    0,
    140,
    140,
    140,
    255,
    255,
    255,
] + [0, 0, 0] * 253

DEFAULT_BIG_FRACTION = 0.5006
DEFAULT_SMALL_FRACTION = 0.2705
TRAIN_EDGE_SLICES = 8

Sphere = tuple[np.ndarray, int, int]


def generate_sphere_geometry(
    *,
    size: int,
    big_radius: int,
    big_fraction: float,
    small_fraction: float,
    seed: int,
) -> tuple[np.ndarray, list[Sphere]]:
    """Generate non-overlapping spheres: background=0, small=1, big=2."""
    _validate_settings(
        size=size,
        big_radius=big_radius,
        big_fraction=big_fraction,
        small_fraction=small_fraction,
        seed=seed,
    )
    small_radius = big_radius // 2
    rng = np.random.default_rng(seed)
    volume = np.zeros((size, size, size), dtype=np.uint8)
    spheres: list[Sphere] = []

    _pour_sphere_mixture(
        volume,
        radii={2: big_radius},
        target_counts={2: round(big_fraction * volume.size)},
        spheres=spheres,
        rng=rng,
    )
    _infill_random_voids(
        volume,
        radius=small_radius,
        label=1,
        target_count=round(small_fraction * volume.size),
        spheres=spheres,
        rng=rng,
    )
    return volume, spheres


def _infill_random_voids(
    volume: np.ndarray,
    *,
    radius: int,
    label: int,
    target_count: int,
    spheres: list[Sphere],
    rng: np.random.Generator,
    max_failed_attempts: int = 20_000,
) -> int:
    """Fill interior 3D voids with collision-free small spheres."""
    filled = int(np.count_nonzero(volume == label))
    failed_attempts = 0
    centers = np.stack([center for center, _, _ in spheres])
    placed_radii = np.asarray(
        [placed_radius for _, placed_radius, _ in spheres],
        dtype=np.int64,
    )

    while filled < target_count and failed_attempts < max_failed_attempts:
        center = rng.integers(
            radius,
            volume.shape[0] - radius,
            size=3,
            dtype=np.int64,
        )
        distance_squared = np.sum((centers - center) ** 2, axis=1)
        minimum_distance_squared = (placed_radii + radius) ** 2
        if np.any(distance_squared <= minimum_distance_squared):
            failed_attempts += 1
            continue

        filled += _paint_sphere(volume, center, radius, label)
        spheres.append((center.copy(), radius, label))
        centers = np.vstack((centers, center))
        placed_radii = np.append(placed_radii, radius)
        failed_attempts = 0
    return filled


def _pour_sphere_mixture(
    volume: np.ndarray,
    *,
    radii: dict[int, int],
    target_counts: dict[int, int],
    spheres: list[Sphere],
    rng: np.random.Generator,
    max_failed_attempts: int = 2_000,
) -> dict[int, int]:
    """Pour a random sphere mixture downward until it settles or jams."""
    filled = {
        label: int(np.count_nonzero(volume == label)) for label in target_counts
    }
    failed_attempts = 0
    centers = (
        np.stack([center for center, _, _ in spheres])
        if spheres
        else np.empty((0, 3), dtype=np.int64)
    )
    placed_radii = np.asarray(
        [radius for _, radius, _ in spheres],
        dtype=np.int64,
    )
    sphere_voxel_counts = {
        label: _sphere_voxel_count(radius) for label, radius in radii.items()
    }

    while failed_attempts < max_failed_attempts:
        remaining_labels = [
            label
            for label, target_count in target_counts.items()
            if filled[label] < target_count
        ]
        if not remaining_labels:
            break
        remaining_spheres = np.asarray(
            [
                (target_counts[label] - filled[label])
                / sphere_voxel_counts[label]
                for label in remaining_labels
            ],
            dtype=np.float64,
        )
        label = int(
            rng.choice(
                remaining_labels,
                p=remaining_spheres / remaining_spheres.sum(),
            )
        )
        radius = radii[label]
        center = _drop_center(
            size=volume.shape[0],
            radius=radius,
            centers=centers,
            placed_radii=placed_radii,
            rng=rng,
        )
        if center is None:
            failed_attempts += 1
            continue

        filled[label] += _paint_sphere(volume, center, radius, label)
        spheres.append((center.copy(), radius, label))
        centers = np.vstack((centers, center))
        placed_radii = np.append(placed_radii, radius)
        failed_attempts = 0
    return filled


def _drop_center(
    *,
    size: int,
    radius: int,
    centers: np.ndarray,
    placed_radii: np.ndarray,
    rng: np.random.Generator,
    roll_attempts: int = 8,
) -> np.ndarray | None:
    """Drop at random X/Y positions and roll toward the deepest result."""
    best_center = None
    for _ in range(roll_attempts):
        y, x = rng.integers(radius, size - radius, size=2, dtype=np.int64)
        settled_z = _settled_z(
            size=size,
            radius=radius,
            y=int(y),
            x=int(x),
            centers=centers,
            placed_radii=placed_radii,
        )
        if settled_z >= radius and (
            best_center is None or settled_z > best_center[0]
        ):
            best_center = np.asarray((settled_z, y, x), dtype=np.int64)
    return best_center


def _settled_z(
    *,
    size: int,
    radius: int,
    y: int,
    x: int,
    centers: np.ndarray,
    placed_radii: np.ndarray,
) -> int:
    settled_z = size - radius - 1
    if centers.size:
        horizontal_squared = (centers[:, 1] - y) ** 2 + (
            centers[:, 2] - x
        ) ** 2
        minimum_distance_squared = (placed_radii + radius) ** 2
        crossing = horizontal_squared <= minimum_distance_squared
        if np.any(crossing):
            vertical_clearance = np.floor(
                np.sqrt(
                    minimum_distance_squared[crossing]
                    - horizontal_squared[crossing]
                )
            ).astype(np.int64) + 1
            settled_z = min(
                settled_z,
                int(np.min(centers[crossing, 0] - vertical_clearance)),
            )
    return settled_z


def generate_sphere_volume(**settings) -> np.ndarray:
    return generate_sphere_geometry(**settings)[0]


def generate_sphere_dataset(
    data_dir: str | Path,
    *,
    size: int,
    big_radius: int,
    big_fraction: float,
    small_fraction: float,
    seed: int,
) -> tuple[Path, list[Path]]:
    """Save one 3D TIFF and its interior Z-slice PNGs."""
    gt_dir, train_dir = _prepare_output_dirs(data_dir, size=size)

    volume = generate_sphere_volume(
        size=size,
        big_radius=big_radius,
        big_fraction=big_fraction,
        small_fraction=small_fraction,
        seed=seed,
    )
    return _save_volume(
        volume,
        gt_dir=gt_dir,
        train_dir=train_dir,
        volume_stem="volume",
        slice_stem="slice",
    )


def generate_sphere_datasets(
    data_dir: str | Path,
    *,
    num_volumes: int,
    size: int,
    big_radius: int,
    big_fraction: float,
    small_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Save multiple independently seeded TIFF volumes and their slices."""
    _require_int("num_volumes", num_volumes)
    if num_volumes <= 0:
        raise ValueError("num_volumes must be positive.")
    gt_dir, train_dir = _prepare_output_dirs(data_dir, size=size)

    volume_paths = []
    slice_paths = []
    for index in range(num_volumes):
        volume = generate_sphere_volume(
            size=size,
            big_radius=big_radius,
            big_fraction=big_fraction,
            small_fraction=small_fraction,
            seed=seed + index,
        )
        stem = f"volume_{index:03d}"
        volume_path, current_slices = _save_volume(
            volume,
            gt_dir=gt_dir,
            train_dir=train_dir,
            volume_stem=stem,
            slice_stem=stem,
        )
        volume_paths.append(volume_path)
        slice_paths.extend(current_slices)
    return volume_paths, slice_paths


def _prepare_output_dirs(
    data_dir: str | Path,
    *,
    size: int,
) -> tuple[Path, Path]:
    data = Path(data_dir)
    gt_dir = data / "gt"
    train_dir = data / "train"
    if size <= 2 * TRAIN_EDGE_SLICES:
        raise ValueError(
            f"size must exceed {2 * TRAIN_EDGE_SLICES} to trim train slices."
        )
    for directory in (gt_dir, train_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"output directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    return gt_dir, train_dir


def _save_volume(
    volume: np.ndarray,
    *,
    gt_dir: Path,
    train_dir: Path,
    volume_stem: str,
    slice_stem: str,
) -> tuple[Path, list[Path]]:
    volume_path = gt_dir / f"{volume_stem}.tif"

    tifffile.imwrite(
        volume_path,
        volume,
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

    slice_paths = []
    train_stop = volume.shape[0] - TRAIN_EDGE_SLICES
    for z in range(TRAIN_EDGE_SLICES, train_stop):
        labels = volume[z]
        slice_path = train_dir / f"{slice_stem}_z_{z:03d}.png"
        image = Image.fromarray(labels)
        image.putpalette(PALETTE)
        image.save(slice_path)
        slice_paths.append(slice_path)
    return volume_path, slice_paths


def _sphere_voxel_count(radius: int) -> int:
    z, y, x = np.ogrid[
        -radius : radius + 1,
        -radius : radius + 1,
        -radius : radius + 1,
    ]
    return int((z**2 + y**2 + x**2 <= radius**2).sum())


def _paint_sphere(
    volume: np.ndarray,
    center: np.ndarray,
    radius: int,
    label: int,
) -> int:
    lower = np.maximum(center - radius, 0)
    upper = np.minimum(center + radius + 1, volume.shape)
    z, y, x = np.ogrid[
        lower[0] - center[0] : upper[0] - center[0],
        lower[1] - center[1] : upper[1] - center[1],
        lower[2] - center[2] : upper[2] - center[2],
    ]
    sphere = z**2 + y**2 + x**2 <= radius**2
    block = volume[
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ]
    available = sphere & (block == 0)
    block[available] = label
    return int(available.sum())


def _validate_settings(
    *,
    size: int,
    big_radius: int,
    big_fraction: float,
    small_fraction: float,
    seed: int,
) -> None:
    for name, value in (
        ("size", size),
        ("big_radius", big_radius),
        ("seed", seed),
    ):
        _require_int(name, value)
    if size < 8:
        raise ValueError("size must be at least 8.")
    if big_radius <= 1 or big_radius >= size / 2 or big_radius % 2 != 0:
        raise ValueError(
            "big_radius must be even and satisfy 1 < big_radius < size / 2."
        )
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    for name, value in (
        ("big_fraction", big_fraction),
        ("small_fraction", small_fraction),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < value < 1.0
        ):
            raise ValueError(f"{name} must be between 0 and 1.")
    if big_fraction + small_fraction >= 1.0:
        raise ValueError("big_fraction + small_fraction must be below 1.")


def _require_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate non-overlapping 3D sphere volumes and their slices."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-volumes", type=int, default=1)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--big-radius", type=int, default=20)
    parser.add_argument(
        "--big-fraction",
        type=float,
        default=DEFAULT_BIG_FRACTION,
    )
    parser.add_argument(
        "--small-fraction",
        type=float,
        default=DEFAULT_SMALL_FRACTION,
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    settings = {
        "size": args.size,
        "big_radius": args.big_radius,
        "big_fraction": args.big_fraction,
        "small_fraction": args.small_fraction,
        "seed": args.seed,
    }
    if args.num_volumes == 1:
        volume_path, slice_paths = generate_sphere_dataset(
            args.data_dir,
            **settings,
        )
        volume_paths = [volume_path]
    else:
        volume_paths, slice_paths = generate_sphere_datasets(
            args.data_dir,
            num_volumes=args.num_volumes,
            **settings,
        )
    print(f"Generated {len(volume_paths)} ground-truth volumes at {volume_paths[0].parent}")
    print(f"Generated {len(slice_paths)} training slices at {slice_paths[0].parent}")


if __name__ == "__main__":
    main()
