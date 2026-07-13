from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.guidance.joint.model import PatchDiscriminator
from src.pipelines.guidance.joint.slices import (
    periodic_shift,
    probability_plane,
    straight_through_one_hot,
    volume_plane,
)


def texture_patches(
    images: torch.Tensor,
    *,
    pool_size: int,
    kernel_size: int,
    flatten: bool = True,
) -> torch.Tensor:
    if pool_size > 1:
        images = F.avg_pool2d(images, kernel_size=pool_size, stride=pool_size)
    patches = F.unfold(
        images,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
    ).transpose(1, 2)
    return patches.reshape(-1, patches.shape[-1]) if flatten else patches


def texture_loss(
    fake_images: torch.Tensor,
    targets: list[tuple[torch.Tensor, torch.Tensor, int, int]] | None,
) -> torch.Tensor:
    if not targets:
        raise ValueError("texture targets are required when texture guidance is enabled.")
    losses = []
    for target_patches, projection, pool_size, kernel_size in targets:
        fake_batches = texture_patches(
            fake_images,
            pool_size=pool_size,
            kernel_size=kernel_size,
            flatten=False,
        )
        for fake_patches in fake_batches:
            count = min(512, int(fake_patches.shape[0]), int(target_patches.shape[0]))
            fake_indices = torch.randperm(fake_patches.shape[0], device=fake_patches.device)[
                :count
            ]
            target_indices = torch.randperm(
                target_patches.shape[0], device=target_patches.device
            )[:count]
            fake_projection = (fake_patches[fake_indices] @ projection).sort(dim=0).values
            target_projection = (
                target_patches[target_indices] @ projection
            ).sort(dim=0).values
            losses.append((fake_projection - target_projection).abs().mean())
    return torch.stack(losses).mean()


def axis_transition_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target is None:
        raise ValueError("transition target is required when transition guidance is enabled.")
    categorical = straight_through_one_hot(probabilities)
    rates = []
    for dimension in (2, 3, 4):
        length = int(categorical.shape[dimension])
        before = categorical.narrow(dimension, 0, length - 1)
        after = categorical.narrow(dimension, 1, length - 1)
        rates.append(1.0 - (before * after).sum(dim=1).mean())
    axis_rates = torch.stack(rates)
    return F.mse_loss(axis_rates, target.expand_as(axis_rates)), axis_rates


def interface_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor | None,
) -> torch.Tensor:
    if target is None:
        raise ValueError("interface target is required when interface guidance is enabled.")
    actual = phase_interface_matrices(probabilities)
    expected = target.unsqueeze(0).expand_as(actual)
    num_phases = int(actual.shape[1])
    pair_loss = F.mse_loss(actual, expected) * num_phases**2
    transition_loss = F.mse_loss(
        actual.sum(dim=(1, 2)),
        expected.sum(dim=(1, 2)),
    )
    return pair_loss + transition_loss


def phase_interface_matrices(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 4:
        raise ValueError("phase probabilities must have shape [B, P, H, W].")
    _, num_phases, height, width = probabilities.shape
    if height < 2 or width < 2:
        raise ValueError("phase probabilities must be at least 2 by 2.")
    horizontal = torch.einsum(
        "bihw,bjhw->bij",
        probabilities[:, :, :, :-1],
        probabilities[:, :, :, 1:],
    ) / (height * (width - 1))
    vertical = torch.einsum(
        "bihw,bjhw->bij",
        probabilities[:, :, :-1, :],
        probabilities[:, :, 1:, :],
    ) / ((height - 1) * width)
    symmetric = 0.25 * (
        horizontal + horizontal.transpose(1, 2) + vertical + vertical.transpose(1, 2)
    )
    off_diagonal = 1.0 - torch.eye(
        num_phases,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return symmetric * off_diagonal.unsqueeze(0)


def update_discriminator(
    discriminator: PatchDiscriminator,
    optimizer: torch.optim.Optimizer,
    fake_probabilities: torch.Tensor,
    real_images: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if real_images is None:
        return {}
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    discriminator.train()
    optimizer.zero_grad()
    batch_size = min(8, int(fake_probabilities.shape[0]), int(real_images.shape[0]))
    real_indices = torch.randint(
        0,
        real_images.shape[0],
        (batch_size,),
        device=real_images.device,
    )
    real = periodic_shift(real_images[real_indices])
    fake = fake_probabilities.detach()
    fake_indices = torch.randperm(fake.shape[0], device=fake.device)[:batch_size]
    real_score = discriminator(real)
    fake_score = discriminator(fake[fake_indices])
    loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
    loss.backward()
    optimizer.step()
    return {
        "patch_discriminator": loss.detach(),
        "patch_real": real_score.mean().detach(),
        "patch_fake": fake_score.mean().detach(),
        "patch_margin": (real_score.mean() - fake_score.mean()).detach(),
    }


def anchor_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    anchors: Sequence[AnchorSlice],
    *,
    radius: int,
    slab_weight: float,
) -> torch.Tensor:
    losses = []
    weights = []
    size = int(target.shape[0])
    for anchor in anchors:
        anchor_target = volume_plane(target, int(anchor.axis), int(anchor.index))
        for offset in range(-radius, radius + 1):
            index = int(anchor.index) + offset
            if index < 0 or index >= size:
                continue
            weight = 1.0 if offset == 0 else slab_weight * (
                radius + 1 - abs(offset)
            ) / (radius + 1)
            if weight == 0.0:
                continue
            plane = probability_plane(probabilities, int(anchor.axis), index)
            losses.append(
                F.nll_loss(
                    plane.clamp_min(torch.finfo(plane.dtype).tiny).log().unsqueeze(0),
                    anchor_target.round().to(torch.long).unsqueeze(0),
                )
            )
            weights.append(weight)
    if losses:
        weight_tensor = probabilities.new_tensor(weights)
        return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()
    active = mask > 0
    selected = probabilities.permute(1, 2, 3, 0)[active]
    indices = target[active].round().to(torch.long)
    return F.nll_loss(
        selected.clamp_min(torch.finfo(selected.dtype).tiny).log(),
        indices,
    )


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
