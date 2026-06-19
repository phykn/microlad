from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "ttiff")
TIFF_EXTENSIONS = ("tif", "tiff", "ttiff")


def _to_uint8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
    if not values.size:
        return values.astype(np.uint8)

    low = float(values.min())
    high = float(values.max())
    if 0.0 <= low and high <= 1.0:
        values = values * 255.0
    elif not (0.0 <= low and high <= 255.0):
        if high == low:
            values = np.full(
                values.shape, 255.0 if high > 0.0 else 0.0, dtype=np.float32
            )
        else:
            values = (values - low) / (high - low) * 255.0

    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _to_gray(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 4 and array.shape[-1] in (1, 3, 4):
        if array.shape[-1] == 1:
            return array[..., 0]
        rgb = array[..., :3].astype(np.float32)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if array.ndim == 4 and array.shape[1] in (1, 3, 4):
        if array.shape[1] == 1:
            return array[:, 0]
        rgb = array[:, :3].astype(np.float32)
        return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    if array.ndim != 3:
        raise ValueError(
            "image data must have shape HxW, HxWxC, CxHxW, DxHxW, DxHxWxC, or DxCxHxW."
        )

    if array.shape[-1] == 1:
        return array[..., 0]
    if array.shape[-1] in (3, 4):
        rgb = array[..., :3].astype(np.float32)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if array.shape[0] == 1:
        return array[0]
    if array.shape[0] in (3, 4):
        rgb = array[:3].astype(np.float32)
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

    return array


def _is_rgb(page: tifffile.TiffPage) -> bool:
    return getattr(page, "samplesperpixel", 1) in (3, 4) and int(page.photometric) == 2


def load_image(path: str | Path) -> np.ndarray:
    """Return image data as uint8 HxW or DxHxW."""
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    if suffix in TIFF_EXTENSIONS:
        with tifffile.TiffFile(path) as tif:
            array = tif.asarray()
            if _is_rgb(tif.pages[0]):
                array = _to_gray(np.asarray(array))
        return _to_uint8(np.asarray(array))

    with Image.open(path) as image:
        array = np.asarray(image)
    return _to_uint8(_to_gray(np.asarray(array)))
