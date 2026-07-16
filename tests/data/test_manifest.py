import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import load_axis_manifest


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)


def _write_manifest(root: Path, sources: dict[str, str]) -> Path:
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "volume_axes": "ZYX",
                "axis_sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loads_sources_relative_to_manifest_not_cwd(tmp_path: Path) -> None:
    for plane in ("xy", "xz", "yz"):
        _write_image(tmp_path / "train" / plane / f"{plane}.png")
    manifest = _write_manifest(
        tmp_path,
        {"yz": "train/yz", "xy": "train/xy", "xz": "train/xz"},
    )
    other = tmp_path / "other"
    other.mkdir()
    previous = Path.cwd()
    try:
        os.chdir(other)
        paths, conditions = load_axis_manifest(manifest)
    finally:
        os.chdir(previous)

    assert [path.parent.name for path in paths] == ["xy", "xz", "yz"]
    assert conditions == (0, 1, 2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("volume_axes", "XYZ", "volume_axes"),
        ("axis_sources", {"xy": "train/xy"}, "xy, xz, and yz"),
    ],
)
def test_rejects_invalid_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "volume_axes": "ZYX",
        "axis_sources": {"xy": "train", "xz": "train", "yz": "train"},
    }
    manifest[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_axis_manifest(path)


@pytest.mark.parametrize("source", ["../outside", "C:/outside", "train\\xy"])
def test_rejects_non_local_or_non_posix_sources(
    tmp_path: Path,
    source: str,
) -> None:
    path = _write_manifest(
        tmp_path,
        {"xy": source, "xz": source, "yz": source},
    )

    with pytest.raises(ValueError, match="source"):
        load_axis_manifest(path)


def test_rejects_empty_source(tmp_path: Path) -> None:
    for plane in ("xy", "xz", "yz"):
        (tmp_path / plane).mkdir()
    path = _write_manifest(
        tmp_path,
        {"xy": "xy", "xz": "xz", "yz": "yz"},
    )

    with pytest.raises(ValueError, match="no supported images"):
        load_axis_manifest(path)
