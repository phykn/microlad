import torch
import torch.nn.functional as F


def axis_loss(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 5 or probabilities.shape[0] < 2:
        raise ValueError(
            "axis probabilities must have shape [axes, phases, depth, height, width]."
        )
    if not probabilities.is_floating_point() or not torch.isfinite(
        probabilities
    ).all():
        raise ValueError("axis probabilities must be finite floating-point values.")
    spatial = tuple(range(2, probabilities.ndim))
    mass = probabilities.sum(dim=spatial, keepdim=True)
    tiny = torch.finfo(probabilities.dtype).tiny
    normalized = probabilities / mass.clamp_min(tiny)
    mean = normalized.mean(dim=0)
    divergence = normalized * (
        normalized.clamp_min(tiny).log() - mean.clamp_min(tiny).log()
    )
    return divergence.sum(dim=spatial).mean()


def axis_mass_loss(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 5 or probabilities.shape[0] < 2:
        raise ValueError(
            "axis probabilities must have shape [axes, phases, depth, height, width]."
        )
    if not probabilities.is_floating_point() or not torch.isfinite(
        probabilities
    ).all():
        raise ValueError("axis probabilities must be finite floating-point values.")

    spatial = tuple(range(2, probabilities.ndim))
    fractions = probabilities.mean(dim=spatial)
    return (fractions - fractions.mean(dim=0, keepdim=True)).square().mean()


def anchor_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active = mask > 0
    selected = probabilities.permute(1, 2, 3, 0)[active]
    indices = target[active].round().to(torch.long)
    pixel_loss = (
        -selected.clamp_min(torch.finfo(selected.dtype).tiny)
        .log()
        .gather(1, indices.unsqueeze(1))[:, 0]
    )
    return torch.stack(
        [pixel_loss[indices == phase].mean() for phase in indices.unique()]
    ).mean()


def continuity_loss(probabilities: torch.Tensor) -> torch.Tensor:
    if min(probabilities.shape[2:]) < 3:
        return probabilities.sum() * 0.0
    smoothed = F.avg_pool3d(
        probabilities,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False,
    )
    curvature = []
    for dimension in (2, 3, 4):
        length = int(smoothed.shape[dimension])
        for lag in (1, 2, 3):
            span = length - 2 * lag
            if span <= 0:
                continue
            before = smoothed.narrow(dimension, 0, span)
            middle = smoothed.narrow(dimension, lag, span)
            after = smoothed.narrow(dimension, 2 * lag, span)
            curvature.append((after - 2.0 * middle + before).abs().mean())
    return torch.stack(curvature).mean()


def fraction_loss(
    actual: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if actual.ndim != 1 or actual.shape != target.shape:
        raise ValueError("actual and target fractions must have the same 1D shape.")
    if not actual.is_floating_point() or not target.is_floating_point():
        raise ValueError("actual and target fractions must be floating point.")
    if not torch.isfinite(actual).all() or not torch.isfinite(target).all():
        raise ValueError("actual and target fractions must be finite.")
    if torch.any(actual < 0.0) or torch.any(target < 0.0):
        raise ValueError("actual and target fractions must be non-negative.")

    epsilon = torch.finfo(actual.dtype).eps
    smoothed = actual + epsilon
    smoothed = smoothed / smoothed.sum()
    return F.kl_div(smoothed.log(), target, reduction="sum")
