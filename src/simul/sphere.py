import math
from numbers import Real

import numpy as np

from ..misc import require_int


Sphere = tuple[np.ndarray, int, int]


def make_geometry(
    *,
    size: int,
    big_radius: int,
    big_fraction: float,
    small_fraction: float,
    seed: int,
) -> tuple[np.ndarray, list[Sphere]]:
    """Create non-overlapping spheres labeled as background, small, and big."""
    _validate(size, big_radius, big_fraction, small_fraction, seed)
    rng = np.random.default_rng(seed)
    vol = np.zeros((size,) * 3, dtype=np.uint8)
    spheres: list[Sphere] = []

    _pour(
        vol,
        radii={2: big_radius},
        targets={2: round(big_fraction * vol.size)},
        spheres=spheres,
        rng=rng,
    )
    _fill_voids(
        vol,
        radius=big_radius // 2,
        label=1,
        target=round(small_fraction * vol.size),
        spheres=spheres,
        rng=rng,
    )
    return vol, spheres


def make_volume(
    *,
    size: int,
    big_radius: int,
    big_fraction: float,
    small_fraction: float,
    seed: int,
) -> np.ndarray:
    return make_geometry(
        size=size,
        big_radius=big_radius,
        big_fraction=big_fraction,
        small_fraction=small_fraction,
        seed=seed,
    )[0]


def _fill_voids(
    vol: np.ndarray,
    *,
    radius: int,
    label: int,
    target: int,
    spheres: list[Sphere],
    rng: np.random.Generator,
    max_fails: int = 20_000,
) -> int:
    filled = int(np.count_nonzero(vol == label))
    fails = 0
    centers = np.stack([center for center, _, _ in spheres])
    radii = np.asarray([radius for _, radius, _ in spheres], dtype=np.int64)

    while filled < target and fails < max_fails:
        center = rng.integers(radius, vol.shape[0] - radius, size=3)
        distance = np.sum((centers - center) ** 2, axis=1)
        if np.any(distance <= (radii + radius) ** 2):
            fails += 1
            continue

        filled += _paint(vol, center, radius, label)
        spheres.append((center.copy(), radius, label))
        centers = np.vstack((centers, center))
        radii = np.append(radii, radius)
        fails = 0
    return filled


def _pour(
    vol: np.ndarray,
    *,
    radii: dict[int, int],
    targets: dict[int, int],
    spheres: list[Sphere],
    rng: np.random.Generator,
    max_fails: int = 2_000,
) -> dict[int, int]:
    filled = {label: int(np.count_nonzero(vol == label)) for label in targets}
    centers = np.empty((0, 3), dtype=np.int64)
    placed = np.empty(0, dtype=np.int64)
    counts = {label: _count(radius) for label, radius in radii.items()}
    fails = 0

    while fails < max_fails:
        labels = [label for label in targets if filled[label] < targets[label]]
        if not labels:
            break

        remaining = np.asarray(
            [(targets[label] - filled[label]) / counts[label] for label in labels]
        )
        label = int(rng.choice(labels, p=remaining / remaining.sum()))
        radius = radii[label]
        center = _drop(vol.shape[0], radius, centers, placed, rng)
        if center is None:
            fails += 1
            continue

        filled[label] += _paint(vol, center, radius, label)
        spheres.append((center.copy(), radius, label))
        centers = np.vstack((centers, center))
        placed = np.append(placed, radius)
        fails = 0
    return filled


def _drop(
    size: int,
    radius: int,
    centers: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator,
    attempts: int = 8,
) -> np.ndarray | None:
    best = None
    for _ in range(attempts):
        y, x = rng.integers(radius, size - radius, size=2)
        z = _settle(size, radius, int(y), int(x), centers, radii)
        if z >= radius and (best is None or z > best[0]):
            best = np.asarray((z, y, x), dtype=np.int64)
    return best


def _settle(
    size: int,
    radius: int,
    y: int,
    x: int,
    centers: np.ndarray,
    radii: np.ndarray,
) -> int:
    z = size - radius - 1
    if not centers.size:
        return z

    horizontal = (centers[:, 1] - y) ** 2 + (centers[:, 2] - x) ** 2
    distance = (radii + radius) ** 2
    crossing = horizontal <= distance
    if crossing.any():
        clearance = (
            np.floor(np.sqrt(distance[crossing] - horizontal[crossing])).astype(
                np.int64
            )
            + 1
        )
        z = min(z, int(np.min(centers[crossing, 0] - clearance)))
    return z


def _count(radius: int) -> int:
    z, y, x = np.ogrid[
        -radius : radius + 1,
        -radius : radius + 1,
        -radius : radius + 1,
    ]
    return int((z**2 + y**2 + x**2 <= radius**2).sum())


def _paint(vol: np.ndarray, center: np.ndarray, radius: int, label: int) -> int:
    low = np.maximum(center - radius, 0)
    high = np.minimum(center + radius + 1, vol.shape)
    z, y, x = np.ogrid[
        low[0] - center[0] : high[0] - center[0],
        low[1] - center[1] : high[1] - center[1],
        low[2] - center[2] : high[2] - center[2],
    ]
    sphere = z**2 + y**2 + x**2 <= radius**2
    block = vol[low[0] : high[0], low[1] : high[1], low[2] : high[2]]
    empty = sphere & (block == 0)
    block[empty] = label
    return int(empty.sum())


def _validate(
    size: int,
    big_radius: int,
    big_fraction: float,
    small_fraction: float,
    seed: int,
) -> None:
    for name, value in (("size", size), ("big_radius", big_radius), ("seed", seed)):
        require_int(name, value)
    if size < 8:
        raise ValueError("size must be at least 8.")
    if big_radius <= 1 or big_radius >= size / 2 or big_radius % 2:
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
        raise ValueError("phase fractions must sum to less than one.")
