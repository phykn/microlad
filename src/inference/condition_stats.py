from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.loss.generate import (
    build_grayscale_tpc_targets,
    compute_relative_surface_area,
    soft_gray_level_masks,
)

from .diffusivity import DiffusivitySolver


@dataclass
class ConditionStats:
    gray_moments: tuple[float, float] | None
    grayscale_tpc_target: torch.Tensor | None
    grayscale_tpc_bin_mat: torch.Tensor | None
    grayscale_tpc_bin_counts: torch.Tensor | None
    surface_area_targets: dict[int, float] | None
    diffusivity_targets: dict[int, float] | None
    diffusivity_solver: torch.nn.Module | None


def empty_condition_stats() -> ConditionStats:
    return ConditionStats(
        gray_moments=None,
        grayscale_tpc_target=None,
        grayscale_tpc_bin_mat=None,
        grayscale_tpc_bin_counts=None,
        surface_area_targets=None,
        diffusivity_targets=None,
        diffusivity_solver=None,
    )


def _build_gray_stats(
    images: list[torch.Tensor],
    gray_levels: list[int],
) -> tuple[
    tuple[float, float],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[int, float],
]:
    means = torch.stack([image.mean() for image in images])
    squared_means = torch.stack([(image**2).mean() for image in images])
    gray_moments = (float(means.mean()), float(squared_means.mean()))

    grayscale_tpc_target, grayscale_tpc_bin_mat, grayscale_tpc_bin_counts = (
        build_grayscale_tpc_targets(images)
    )

    surface_area_values = [
        compute_relative_surface_area(image, gray_levels) for image in images
    ]
    surface_area_mean = torch.stack(surface_area_values).mean(dim=0)
    surface_area_targets = {
        level: float(surface_area_mean[index])
        for index, level in enumerate(gray_levels)
    }

    return (
        gray_moments,
        grayscale_tpc_target,
        grayscale_tpc_bin_mat,
        grayscale_tpc_bin_counts,
        surface_area_targets,
    )


def _build_diffusivity_stats(
    images: list[torch.Tensor],
    diffusivity_size: int,
    gray_levels: list[int],
    device: torch.device,
) -> tuple[dict[int, float], DiffusivitySolver]:
    height, width = images[0].shape[-2:]
    solver_height = min(height, diffusivity_size)
    solver_width = min(width, diffusivity_size)
    diffusivity_solver = DiffusivitySolver(solver_height, solver_width, device=device)

    image_values = []
    for image in images:
        if image.shape[-2:] != (height, width):
            raise ValueError(
                "all diffusivity condition images must have the same image shape."
            )
        masks = soft_gray_level_masks(image, gray_levels)
        if masks.shape[-2:] != (solver_height, solver_width):
            masks = F.interpolate(
                masks,
                size=(solver_height, solver_width),
                mode="bilinear",
                align_corners=False,
            )
        image_values.append(
            torch.stack(
                [
                    diffusivity_solver(masks[0, level_index])
                    for level_index in range(len(gray_levels))
                ]
            )
        )

    mean_values = torch.stack(image_values).mean(dim=0)
    diffusivity_targets = {
        level: float(mean_values[index]) for index, level in enumerate(gray_levels)
    }
    return diffusivity_targets, diffusivity_solver


def build_condition_stats(
    condition_images: list[torch.Tensor] | None,
    stats_weight: float,
    diffusivity_weight: float,
    diffusivity_size: int,
    gray_levels: list[int],
    device: torch.device,
) -> ConditionStats:
    if stats_weight < 0:
        raise ValueError("stats_weight must be non-negative.")
    if diffusivity_weight < 0:
        raise ValueError("diffusivity_weight must be non-negative.")
    if diffusivity_size <= 0:
        raise ValueError("diffusivity_size must be positive.")

    if stats_weight <= 0 and diffusivity_weight <= 0:
        return empty_condition_stats()
    if not condition_images:
        raise ValueError(
            "stats_weight or diffusivity_weight requires image-space condition input."
        )

    images = [
        image.to(device=device, dtype=torch.float32) for image in condition_images
    ]
    gray_moments = None
    grayscale_tpc_target = None
    grayscale_tpc_bin_mat = None
    grayscale_tpc_bin_counts = None
    surface_area_targets = None
    diffusivity_targets = None
    diffusivity_solver = None

    if stats_weight > 0:
        (
            gray_moments,
            grayscale_tpc_target,
            grayscale_tpc_bin_mat,
            grayscale_tpc_bin_counts,
            surface_area_targets,
        ) = _build_gray_stats(images, gray_levels)

    if diffusivity_weight > 0:
        diffusivity_targets, diffusivity_solver = _build_diffusivity_stats(
            images=images,
            diffusivity_size=diffusivity_size,
            gray_levels=gray_levels,
            device=device,
        )

    return ConditionStats(
        gray_moments=gray_moments,
        grayscale_tpc_target=grayscale_tpc_target,
        grayscale_tpc_bin_mat=grayscale_tpc_bin_mat,
        grayscale_tpc_bin_counts=grayscale_tpc_bin_counts,
        surface_area_targets=surface_area_targets,
        diffusivity_targets=diffusivity_targets,
        diffusivity_solver=diffusivity_solver,
    )
