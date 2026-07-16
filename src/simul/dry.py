from dataclasses import dataclass

import numpy as np

from ..misc import require_int, require_number


@dataclass(frozen=True)
class DryParticle:
    """One whole hard particle in ``[z, y, x]`` voxel coordinates."""

    center: tuple[int, int, int]
    axes: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    label: int


@dataclass(frozen=True)
class PackingReport:
    requested_fractions: tuple[float, float, float]
    achieved_fractions: tuple[float, float, float]
    particle_counts: tuple[int, int]
    phase_contact_counts: tuple[int, int, int]
    particle_contacts: int

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_fractions": list(self.requested_fractions),
            "achieved_fractions": list(self.achieved_fractions),
            "particle_counts": {
                "small": self.particle_counts[0],
                "big": self.particle_counts[1],
            },
            "face_contacts": {
                "background_small": self.phase_contact_counts[0],
                "background_big": self.phase_contact_counts[1],
                "small_big": self.phase_contact_counts[2],
                "particle_pairs": self.particle_contacts,
            },
        }


@dataclass(frozen=True)
class DryGeometry:
    labels: np.ndarray
    instances: np.ndarray
    particles: tuple[DryParticle, ...]
    report: PackingReport


def make_dry_geometry(
    *,
    size: int,
    big_radius: int,
    small_radius: int | None = None,
    big_fraction: float,
    small_fraction: float,
    shape: str = "sphere",
    elongation: float = 2.0,
    alignment_axis: str = "z",
) -> DryGeometry:
    """Place whole spheres or aligned ellipsoids without voxel overlap.

    The output is intentionally a simple experimental fixture.  Large particles
    are distributed randomly throughout the box first; small particles then
    fill voids throughout the same box.  Every instance is an unclipped
    analytic primitive, all primitives stay inside the box, and occupied
    voxels have exactly one owner.  Fractions are placement guides, not quotas:
    packing stops when the guide is reached or no valid center remains.
    """

    _validate(
        size=size,
        big_radius=big_radius,
        small_radius=small_radius,
        big_fraction=big_fraction,
        small_fraction=small_fraction,
        shape=shape,
        elongation=elongation,
        alignment_axis=alignment_axis,
    )
    rng = np.random.default_rng()
    resolved_small_radius = (
        float(big_radius) / 2.0
        if small_radius is None
        else float(small_radius)
    )
    volume = np.zeros((size,) * 3, dtype=np.uint8)
    instances = np.full((size,) * 3, -1, dtype=np.int32)
    particles: list[DryParticle] = []
    targets = {
        1: round(float(small_fraction) * volume.size),
        2: round(float(big_fraction) * volume.size),
    }

    # Large particles go first so the small phase fills the remaining space.
    for label in (2, 1):
        _place_phase(
            rng,
            volume=volume,
            instances=instances,
            particles=particles,
            label=label,
            target_voxels=targets[label],
            big_radius=float(big_radius),
            small_radius=resolved_small_radius,
            shape=shape,
            elongation=float(elongation),
            alignment_axis=alignment_axis,
        )

    counts = np.bincount(volume.ravel(), minlength=3)
    achieved = tuple(float(value / volume.size) for value in counts)
    requested = (
        1.0 - float(small_fraction) - float(big_fraction),
        float(small_fraction),
        float(big_fraction),
    )
    small_count = sum(particle.label == 1 for particle in particles)
    big_count = sum(particle.label == 2 for particle in particles)
    report = PackingReport(
        requested_fractions=requested,
        achieved_fractions=achieved,
        particle_counts=(small_count, big_count),
        phase_contact_counts=_phase_contact_counts(volume),
        particle_contacts=_particle_contact_count(instances),
    )
    return DryGeometry(
        labels=volume,
        instances=instances,
        particles=tuple(particles),
        report=report,
    )


def make_dry_volume(**settings) -> np.ndarray:
    return make_dry_geometry(**settings).labels


def _place_phase(
    rng: np.random.Generator,
    *,
    volume: np.ndarray,
    instances: np.ndarray,
    particles: list[DryParticle],
    label: int,
    target_voxels: int,
    big_radius: float,
    small_radius: float,
    shape: str,
    elongation: float,
    alignment_axis: str,
) -> None:
    axes, rotation, offsets = _make_primitive(
        label=label,
        big_radius=big_radius,
        small_radius=small_radius,
        shape=shape,
        elongation=elongation,
        alignment_axis=alignment_axis,
    )
    particle_count = round(target_voxels / len(offsets))
    if particle_count <= 0:
        return

    available = _available_centers(volume.shape[0], offsets)
    world_axes = np.abs(rotation) @ axes
    for particle in particles:
        _invalidate_centers(
            available,
            np.asarray(particle.center),
            world_axes,
            np.abs(np.asarray(particle.rotation)) @ np.asarray(particle.axes),
        )

    # Shuffle once, then progressively remove centers made invalid by each
    # placement. This keeps both phases spread through the full 3D box.
    candidates = rng.permutation(np.flatnonzero(available))
    placed = 0
    for flat_index in candidates:
        if placed >= particle_count:
            break
        if not available.flat[flat_index]:
            continue
        center = np.asarray(
            np.unravel_index(flat_index, available.shape), dtype=np.int32
        )
        coordinates = offsets + center
        key = tuple(coordinates.T)
        if np.any(instances[key] >= 0):
            available.flat[flat_index] = False
            continue

        index = len(particles)
        volume[key] = label
        instances[key] = index
        particles.append(
            DryParticle(
                center=tuple(int(value) for value in center),
                axes=tuple(float(value) for value in axes),
                rotation=tuple(
                    tuple(float(value) for value in row) for row in rotation
                ),
                label=label,
            )
        )
        placed += 1
        _invalidate_centers(available, center, world_axes, world_axes)


def _make_primitive(
    *,
    label: int,
    big_radius: float,
    small_radius: float,
    shape: str,
    elongation: float,
    alignment_axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = big_radius if label == 2 else small_radius

    if shape == "sphere":
        axes = np.full(3, radius, dtype=np.float64)
        rotation = np.eye(3)
    else:
        # Keep primitive volume constant while changing only its aspect ratio.
        short = radius / elongation ** (1.0 / 3.0)
        long = radius * elongation ** (2.0 / 3.0)
        axes = np.asarray((long, short, short), dtype=np.float64)
        rotation = _alignment_rotation(alignment_axis)
    return axes, rotation, _primitive_offsets(axes, rotation)


def _primitive_offsets(axes: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rasterize one complete analytic primitive around the origin."""

    bounds = np.ceil(np.abs(rotation) @ axes).astype(np.int32)
    z, y, x = np.meshgrid(
        np.arange(-bounds[0], bounds[0] + 1, dtype=np.int32),
        np.arange(-bounds[1], bounds[1] + 1, dtype=np.int32),
        np.arange(-bounds[2], bounds[2] + 1, dtype=np.int32),
        indexing="ij",
    )
    offsets = np.column_stack((z.ravel(), y.ravel(), x.ravel()))
    local = offsets @ rotation
    inside = np.sum((local / axes) ** 2, axis=1) <= 1.0 + 1e-12
    return offsets[inside]


def _available_centers(size: int, offsets: np.ndarray) -> np.ndarray:
    """Return centers for which the complete primitive stays inside the box."""

    available = np.zeros((size,) * 3, dtype=bool)
    lower = -offsets.min(axis=0)
    upper = size - 1 - offsets.max(axis=0)
    if np.any(lower > upper):
        return available
    available[
        lower[0] : upper[0] + 1,
        lower[1] : upper[1] + 1,
        lower[2] : upper[2] + 1,
    ] = True
    return available


def _invalidate_centers(
    available: np.ndarray,
    center: np.ndarray,
    candidate_axes: np.ndarray,
    existing_axes: np.ndarray,
) -> None:
    """Remove candidate centers that would overlap one hard particle."""

    combined = candidate_axes + existing_axes
    bounds = np.ceil(combined).astype(np.int32)
    low = np.maximum(center - bounds, 0)
    high = np.minimum(center + bounds + 1, available.shape)
    z, y, x = np.ogrid[
        low[0] - center[0] : high[0] - center[0],
        low[1] - center[1] : high[1] - center[1],
        low[2] - center[2] : high[2] - center[2],
    ]
    overlaps = (
        (z / combined[0]) ** 2
        + (y / combined[1]) ** 2
        + (x / combined[2]) ** 2
        <= 1.0
    )
    block = available[
        low[0] : high[0], low[1] : high[1], low[2] : high[2]
    ]
    block[overlaps] = False


def _alignment_rotation(axis: str) -> np.ndarray:
    axis_index = {"z": 0, "y": 1, "x": 2}[axis]
    long = np.eye(3)[axis_index]
    reference = np.eye(3)[(axis_index + 1) % 3]
    short_two = np.cross(long, reference)
    short_two /= np.linalg.norm(short_two)
    short_one = np.cross(short_two, long)
    return np.column_stack((long, short_one, short_two))


def _phase_contact_counts(volume: np.ndarray) -> tuple[int, int, int]:
    pairs = ((0, 1), (0, 2), (1, 2))
    counts = [0, 0, 0]
    for axis in range(3):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)
        left = volume[tuple(left_slice)]
        right = volume[tuple(right_slice)]
        for index, (first, second) in enumerate(pairs):
            counts[index] += int(
                np.count_nonzero(
                    ((left == first) & (right == second))
                    | ((left == second) & (right == first))
                )
            )
    return tuple(counts)


def _particle_contact_count(instances: np.ndarray) -> int:
    contacts: set[tuple[int, int]] = set()
    for axis in range(3):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)
        left = instances[tuple(left_slice)]
        right = instances[tuple(right_slice)]
        touching = (left >= 0) & (right >= 0) & (left != right)
        for first, second in zip(left[touching], right[touching], strict=True):
            contacts.add(tuple(sorted((int(first), int(second)))))
    return len(contacts)


def _validate(
    *,
    size: int,
    big_radius: int,
    small_radius: int | None,
    big_fraction: float,
    small_fraction: float,
    shape: str,
    elongation: float,
    alignment_axis: str,
) -> None:
    require_int("size", size)
    require_int("big_radius", big_radius)
    if size < 8:
        raise ValueError("size must be at least 8.")
    if big_radius <= 1 or big_radius >= size / 2:
        raise ValueError("big_radius must satisfy 1 < big_radius < size / 2.")
    if small_radius is not None:
        require_int("small_radius", small_radius)
        if small_radius <= 1 or small_radius >= big_radius:
            raise ValueError(
                "small_radius must satisfy 1 < small_radius < big_radius."
            )
    if shape not in {"sphere", "aligned_ellipsoid"}:
        raise ValueError("shape must be 'sphere' or 'aligned_ellipsoid'.")
    if alignment_axis not in {"z", "y", "x"}:
        raise ValueError("alignment_axis must be 'z', 'y', or 'x'.")
    for name, value in (
        ("big_fraction", big_fraction),
        ("small_fraction", small_fraction),
    ):
        require_number(name, value)
        if not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be at least zero and less than one.")
    if big_fraction + small_fraction <= 0.0:
        raise ValueError("at least one phase fraction must be positive.")
    if big_fraction + small_fraction >= 1.0:
        raise ValueError("phase fractions must sum to less than one.")

    for name, value, low, high in (
        ("elongation", elongation, 1.0, 4.0),
    ):
        require_number(name, value)
        number = float(value)
        if not low <= number <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")
