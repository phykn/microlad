from dataclasses import dataclass

import numpy as np

from ..misc import require_int, require_number


@dataclass(frozen=True)
class Particle:
    center: tuple[int, int, int]
    axes: tuple[float, float, float]
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
class Geometry:
    labels: np.ndarray
    instances: np.ndarray
    particles: tuple[Particle, ...]
    report: PackingReport


def make_geometry(
    *,
    size: int,
    big_radius: int,
    small_radius: int | None = None,
    big_fraction: float,
    small_fraction: float,
    big_elongation: float = 1.0,
) -> Geometry:
    _validate(
        size=size,
        big_radius=big_radius,
        small_radius=small_radius,
        big_fraction=big_fraction,
        small_fraction=small_fraction,
        big_elongation=big_elongation,
    )
    rng = np.random.default_rng()
    small_r = (
        float(big_radius) / 2.0
        if small_radius is None
        else float(small_radius)
    )
    work = size * 3 // 2
    vol = np.zeros((work,) * 3, dtype=np.uint8)
    ids = np.full((work,) * 3, -1, dtype=np.int32)
    parts: list[Particle] = []
    target = {
        1: round(float(small_fraction) * vol.size),
        2: round(float(big_fraction) * vol.size),
    }

    # Large particles go first so the small phase fills the remaining space.
    for label in (2, 1):
        _place_phase(
            rng,
            vol=vol,
            ids=ids,
            parts=parts,
            label=label,
            target=target[label],
            big_r=float(big_radius),
            small_r=small_r,
            elong=float(big_elongation),
        )

    # Cropping a larger field removes center-placement bias at output edges.
    vol, ids, parts = _crop(vol, ids, parts, size)
    n = np.bincount(vol.ravel(), minlength=3)
    got = tuple(float(x / vol.size) for x in n)
    want = (
        1.0 - float(small_fraction) - float(big_fraction),
        float(small_fraction),
        float(big_fraction),
    )
    n_small = sum(p.label == 1 for p in parts)
    n_big = sum(p.label == 2 for p in parts)
    stats = PackingReport(
        requested_fractions=want,
        achieved_fractions=got,
        particle_counts=(n_small, n_big),
        phase_contact_counts=_phase_contact_counts(vol),
        particle_contacts=_particle_contact_count(ids),
    )
    return Geometry(
        labels=vol,
        instances=ids,
        particles=tuple(parts),
        report=stats,
    )


def make_volume(**settings) -> np.ndarray:
    return make_geometry(**settings).labels


def _place_phase(
    rng: np.random.Generator,
    *,
    vol: np.ndarray,
    ids: np.ndarray,
    parts: list[Particle],
    label: int,
    target: int,
    big_r: float,
    small_r: float,
    elong: float,
) -> None:
    axes, off = _make_primitive(
        label=label,
        big_r=big_r,
        small_r=small_r,
        elong=elong,
    )
    count = round(target / len(off))
    if count <= 0:
        return

    valid = _available_centers(vol.shape[0], off)
    for part in parts:
        _invalidate_centers(
            valid,
            np.asarray(part.center),
            axes,
            np.asarray(part.axes),
        )

    # Whole-box sampling prevents phase clustering.
    order = rng.permutation(np.flatnonzero(valid))
    n = 0
    for flat in order:
        if n >= count:
            break
        if not valid.flat[flat]:
            continue
        ctr = np.asarray(
            np.unravel_index(flat, valid.shape), dtype=np.int32
        )
        pos = off + ctr
        key = tuple(pos.T)
        if np.any(ids[key] >= 0):
            valid.flat[flat] = False
            continue

        idx = len(parts)
        vol[key] = label
        ids[key] = idx
        parts.append(
            Particle(
                center=tuple(int(x) for x in ctr),
                axes=tuple(float(x) for x in axes),
                label=label,
            )
        )
        n += 1
        _invalidate_centers(valid, ctr, axes, axes)


def _crop(
    vol: np.ndarray,
    ids: np.ndarray,
    parts: list[Particle],
    size: int,
) -> tuple[np.ndarray, np.ndarray, list[Particle]]:
    lo = (vol.shape[0] - size) // 2
    hi = lo + size
    key = (slice(lo, hi),) * 3
    vol = vol[key].copy()
    ids = ids[key].copy()

    used = np.unique(ids[ids >= 0])
    lut = np.full(len(parts), -1, dtype=np.int32)
    lut[used] = np.arange(len(used), dtype=np.int32)
    mask = ids >= 0
    ids[mask] = lut[ids[mask]]

    shift = np.full(3, lo)
    parts = [
        Particle(
            center=tuple(int(x) for x in np.asarray(parts[i].center) - shift),
            axes=parts[i].axes,
            label=parts[i].label,
        )
        for i in used
    ]
    return vol, ids, parts


def _make_primitive(
    *,
    label: int,
    big_r: float,
    small_r: float,
    elong: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = big_r if label == 2 else small_r

    if label == 1:
        axes = np.full(3, r, dtype=np.float64)
    else:
        short = r / elong ** (1.0 / 3.0)
        long = r * elong ** (2.0 / 3.0)
        axes = np.asarray((long, short, short), dtype=np.float64)
    return axes, _primitive_offsets(axes)


def _primitive_offsets(axes: np.ndarray) -> np.ndarray:
    bnd = np.ceil(axes).astype(np.int32)
    z, y, x = np.meshgrid(
        np.arange(-bnd[0], bnd[0] + 1, dtype=np.int32),
        np.arange(-bnd[1], bnd[1] + 1, dtype=np.int32),
        np.arange(-bnd[2], bnd[2] + 1, dtype=np.int32),
        indexing="ij",
    )
    off = np.column_stack((z.ravel(), y.ravel(), x.ravel()))
    keep = np.sum((off / axes) ** 2, axis=1) <= 1.0 + 1e-12
    return off[keep]


def _available_centers(size: int, off: np.ndarray) -> np.ndarray:
    valid = np.zeros((size,) * 3, dtype=bool)
    lo = -off.min(axis=0)
    hi = size - 1 - off.max(axis=0)
    if np.any(lo > hi):
        return valid
    valid[
        lo[0] : hi[0] + 1,
        lo[1] : hi[1] + 1,
        lo[2] : hi[2] + 1,
    ] = True
    return valid


def _invalidate_centers(
    valid: np.ndarray,
    ctr: np.ndarray,
    cand: np.ndarray,
    prev: np.ndarray,
) -> None:
    span = cand + prev
    bnd = np.ceil(span).astype(np.int32)
    lo = np.maximum(ctr - bnd, 0)
    hi = np.minimum(ctr + bnd + 1, valid.shape)
    z, y, x = np.ogrid[
        lo[0] - ctr[0] : hi[0] - ctr[0],
        lo[1] - ctr[1] : hi[1] - ctr[1],
        lo[2] - ctr[2] : hi[2] - ctr[2],
    ]
    hit = (
        (z / span[0]) ** 2
        + (y / span[1]) ** 2
        + (x / span[2]) ** 2
        <= 1.0
    )
    view = valid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    view[hit] = False


def _phase_contact_counts(vol: np.ndarray) -> tuple[int, int, int]:
    pairs = ((0, 1), (0, 2), (1, 2))
    n = [0, 0, 0]
    for ax in range(3):
        ia = [slice(None)] * 3
        ib = [slice(None)] * 3
        ia[ax] = slice(None, -1)
        ib[ax] = slice(1, None)
        a = vol[tuple(ia)]
        b = vol[tuple(ib)]
        for i, (x, y) in enumerate(pairs):
            n[i] += int(
                np.count_nonzero(
                    ((a == x) & (b == y)) | ((a == y) & (b == x))
                )
            )
    return tuple(n)


def _particle_contact_count(ids: np.ndarray) -> int:
    hits: set[tuple[int, int]] = set()
    for ax in range(3):
        ia = [slice(None)] * 3
        ib = [slice(None)] * 3
        ia[ax] = slice(None, -1)
        ib[ax] = slice(1, None)
        a = ids[tuple(ia)]
        b = ids[tuple(ib)]
        mask = (a >= 0) & (b >= 0) & (a != b)
        for x, y in zip(a[mask], b[mask], strict=True):
            hits.add(tuple(sorted((int(x), int(y)))))
    return len(hits)


def _validate(
    *,
    size: int,
    big_radius: int,
    small_radius: int | None,
    big_fraction: float,
    small_fraction: float,
    big_elongation: float,
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

    require_number("big_elongation", big_elongation)
    if not 1.0 <= float(big_elongation) <= 4.0:
        raise ValueError("big_elongation must be between 1.0 and 4.0.")
