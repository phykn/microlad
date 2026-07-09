import math
from numbers import Real
from collections.abc import Sequence

import numpy as np
import torch

from src.predict.sds.diffusivity import DiffusivitySolver, compute_diffusivity
from src.predict.sds.sa import compute_surface_area
from src.predict.sds.tpc import compute_tpc
from src.predict.sds.vf import compute_volume_fraction
from src.predict.types import MAX_UINT8_PHASES
from src.segment import segment_multi_otsu


def build_sds_targets(
    images: Sequence[np.ndarray],
    *,
    num_phases: int,
    segment: bool = False,
    use_vf: bool = False,
    use_tpc: bool = False,
    use_sa: bool = False,
    use_diffusivity: bool = False,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
    diffusivity_size: int | tuple[int, int] | None = None,
    diffusivity_low_cond: float = 0.0,
) -> dict[str, torch.Tensor | DiffusivitySolver]:
    _validate_options(
        num_phases=num_phases,
        temperature=temperature,
        sa_kernel_size=sa_kernel_size,
        sa_sigma=sa_sigma,
        use_diffusivity=use_diffusivity,
        diffusivity_size=diffusivity_size,
        diffusivity_low_cond=diffusivity_low_cond,
    )
    if not _uses_any_target(
        use_vf=use_vf,
        use_tpc=use_tpc,
        use_sa=use_sa,
        use_diffusivity=use_diffusivity,
    ):
        return {}

    values = _prepare_images(
        images,
        num_phases=num_phases,
        segment=segment,
    )
    targets: dict[str, torch.Tensor | DiffusivitySolver] = {}

    if use_vf:
        targets["vf_targets"] = compute_volume_fraction(
            values,
            num_phases=num_phases,
            temperature=temperature,
        ).detach()

    if use_tpc:
        targets["tpc_targets"] = compute_tpc(
            values,
            num_phases=num_phases,
            temperature=temperature,
        ).detach()

    if use_sa:
        targets["sa_targets"] = compute_surface_area(
            values,
            num_phases=num_phases,
            temperature=temperature,
            kernel_size=sa_kernel_size,
            sigma=sa_sigma,
        ).detach()

    if use_diffusivity:
        height, width = _diffusivity_shape(diffusivity_size)
        solver = DiffusivitySolver(
            height=height,
            width=width,
            low_cond=diffusivity_low_cond,
        )
        targets["diffusivity_targets"] = compute_diffusivity(
            values,
            solver=solver,
            num_phases=num_phases,
            temperature=temperature,
        ).detach()
        targets["diffusivity_solver"] = solver

    return targets


def _uses_any_target(
    *,
    use_vf: bool,
    use_tpc: bool,
    use_sa: bool,
    use_diffusivity: bool,
) -> bool:
    return use_vf or use_tpc or use_sa or use_diffusivity


def _prepare_images(
    images: Sequence[np.ndarray],
    *,
    num_phases: int,
    segment: bool,
) -> torch.Tensor:
    if not images:
        raise ValueError("images are required when target options are enabled.")

    phases = []
    shape = None
    for image in images:
        phase = _prepare_phase_image(
            image,
            num_phases=num_phases,
            segment=segment,
        )
        if shape is None:
            shape = phase.shape
        elif phase.shape != shape:
            raise ValueError("target images must have the same shape.")

        phases.append(phase)

    values = np.stack(phases).astype(np.float32)
    return torch.from_numpy(values.copy()).float()


def _prepare_phase_image(
    image: np.ndarray,
    *,
    num_phases: int,
    segment: bool,
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("images must be numpy arrays.")

    if image.ndim != 2:
        raise ValueError("target images must be 2D.")

    if image.size == 0:
        raise ValueError("target images must be non-empty.")

    if image.dtype != np.uint8:
        raise ValueError("target images must have dtype uint8.")

    phase = segment_multi_otsu(image, num_phases) if segment else image

    if phase.min() < 0 or phase.max() >= num_phases:
        raise ValueError(
            f"target images must contain values from 0 to {num_phases - 1}."
        )

    return phase


def _validate_options(
    *,
    num_phases: int,
    temperature: float,
    sa_kernel_size: int,
    sa_sigma: float,
    use_diffusivity: bool,
    diffusivity_size: int | tuple[int, int] | None,
    diffusivity_low_cond: float,
) -> None:
    _validate_integer("num_phases", num_phases)

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2.")

    if num_phases > MAX_UINT8_PHASES:
        raise ValueError(
            f"num_phases must be at most {MAX_UINT8_PHASES} for uint8 images."
        )

    _validate_positive_scalar("temperature", temperature)

    _validate_integer("sa_kernel_size", sa_kernel_size)
    if sa_kernel_size <= 0 or sa_kernel_size % 2 == 0:
        raise ValueError("sa_kernel_size must be a positive odd integer.")

    _validate_positive_scalar("sa_sigma", sa_sigma)

    if use_diffusivity:
        _diffusivity_shape(diffusivity_size)

    _validate_finite_scalar("diffusivity_low_cond", diffusivity_low_cond)
    if diffusivity_low_cond < 0.0 or diffusivity_low_cond > 1.0:
        raise ValueError("diffusivity_low_cond must be between 0 and 1.")


def _diffusivity_shape(size: int | tuple[int, int] | None) -> tuple[int, int]:
    if size is None:
        raise ValueError("diffusivity_size is required when use_diffusivity is True.")

    if isinstance(size, int) and not isinstance(size, bool):
        height = width = size
    elif isinstance(size, tuple) and len(size) == 2:
        height, width = size
    else:
        raise ValueError("diffusivity_size must be an integer or (height, width).")

    _validate_integer("diffusivity_size height", height)
    _validate_integer("diffusivity_size width", width)

    if height < 2:
        raise ValueError("diffusivity_size height must be at least 2.")

    if width < 2:
        raise ValueError("diffusivity_size width must be at least 2.")

    return int(height), int(width)


def _validate_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")


def _validate_positive_scalar(name: str, value: float) -> None:
    _validate_finite_scalar(name, value)

    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _validate_finite_scalar(name: str, value: float) -> None:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar.")

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
