from dataclasses import dataclass

import torch

from src.loss import build_grayscale_tpc_targets, compute_relative_surface_area


@dataclass
class ConditionStats:
    vf_moments: tuple[float, float] | None
    grayscale_tpc_target: torch.Tensor | None
    grayscale_tpc_bin_mat: torch.Tensor | None
    grayscale_tpc_bin_counts: torch.Tensor | None
    sa_targets: dict[int, float] | None


def build_condition_stats(
    condition_images: list[torch.Tensor] | None,
    stats_weight: float,
    phases: list[int],
    device: torch.device,
) -> ConditionStats:
    if stats_weight <= 0:
        return ConditionStats(
            vf_moments=None,
            grayscale_tpc_target=None,
            grayscale_tpc_bin_mat=None,
            grayscale_tpc_bin_counts=None,
            sa_targets=None,
        )
    if not condition_images:
        raise ValueError("stats_weight requires image-space condition input.")

    images = [image.to(device=device, dtype=torch.float32) for image in condition_images]
    means = torch.stack([image.mean() for image in images])
    sqmeans = torch.stack([(image**2).mean() for image in images])
    vf_moments = (float(means.mean()), float(sqmeans.mean()))

    grayscale_tpc_target, grayscale_tpc_bin_mat, grayscale_tpc_bin_counts = build_grayscale_tpc_targets(images)

    sa_values = [compute_relative_surface_area(image, phases) for image in images]
    sa_mean = torch.stack(sa_values).mean(dim=0)
    sa_targets = {phase: float(sa_mean[index]) for index, phase in enumerate(phases)}

    return ConditionStats(
        vf_moments=vf_moments,
        grayscale_tpc_target=grayscale_tpc_target,
        grayscale_tpc_bin_mat=grayscale_tpc_bin_mat,
        grayscale_tpc_bin_counts=grayscale_tpc_bin_counts,
        sa_targets=sa_targets,
    )
