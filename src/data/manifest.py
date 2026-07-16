import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
VOLUME_AXES = "ZYX"
AXIS_PLANES = ("xy", "xz", "yz")
PLANE_AXES = {"xy": (0, "z"), "xz": (1, "y"), "yz": (2, "x")}
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_axis_manifest(
    path: str | Path,
) -> tuple[tuple[Path, ...], tuple[int, ...]]:
    """Load axis image paths relative to the manifest, never the CWD."""

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"axis manifest is required: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"axis manifest is malformed: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("axis manifest must contain a mapping.")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"axis manifest schema_version must be {SCHEMA_VERSION}.")
    if manifest.get("volume_axes") != VOLUME_AXES:
        raise ValueError(f"axis manifest volume_axes must be '{VOLUME_AXES}'.")

    sources = manifest.get("axis_sources")
    if not isinstance(sources, Mapping):
        raise ValueError("axis manifest axis_sources must be a mapping.")
    if set(sources) != set(AXIS_PLANES):
        raise ValueError("axis manifest must define xy, xz, and yz sources.")

    paths: list[Path] = []
    conditions: list[int] = []
    root = manifest_path.parent.resolve()
    for condition, plane in enumerate(AXIS_PLANES):
        directory = _resolve_source(root, sources[plane])
        images = sorted(
            candidate
            for candidate in directory.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(
                f"axis manifest {plane} source contains no supported images: "
                f"{directory}"
            )
        paths.extend(images)
        conditions.extend([condition] * len(images))
    return tuple(paths), tuple(conditions)


def _resolve_source(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(
            "axis manifest source must be a POSIX relative directory."
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(":" in part for part in relative.parts)
    ):
        raise ValueError(
            "axis manifest source must stay inside the manifest directory."
        )
    directory = root.joinpath(*relative.parts).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "axis manifest source must stay inside the manifest directory."
        ) from exc
    if not directory.is_dir():
        raise FileNotFoundError(f"axis manifest source is required: {directory}")
    return directory
