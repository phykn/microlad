from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.modeling.diffusion import DDPMProcess
from src.modeling.phases.representation import (
    phase_levels,
)
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.guidance.descriptors.run_profile import (
    compute_run_profile,
    run_profile_loss,
)
from src.pipelines.guidance.objective import sample_descriptor_loss
from src.pipelines.guidance.physics.diffusivity import DiffusivitySolver
from src.pipelines.guidance.preparation import (
    build_anchor_constraint_volume,
    freeze_inference,
)
from src.pipelines.guidance.prior import sds_loss
from src.pipelines.guidance.target_values import build_phase_target


_AXIS_ORDERS = (
    (0, 1, 2),
    (2, 1, 0),
    (1, 0, 2),
    (2, 0, 1),
    (0, 2, 1),
    (1, 2, 0),
)

_RUN_PROFILE_LENGTHS = (2, 4, 8, 16)


def optimize_joint_volume(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    t_min: int,
    t_max: int,
    num_phases: int,
    anchors: Sequence[AnchorSlice] | None = None,
    anchor_segment: bool = False,
    sds_weight: float = 1.0,
    anchor_weight: float = 0.0,
    anchor_slab_radius: int = 0,
    anchor_slab_weight: float = 0.0,
    vf_targets: Mapping[int, float] | torch.Tensor | None = None,
    vf_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: DiffusivitySolver | None = None,
    diffusivity_weight: float = 0.0,
    entropy_weight: float = 1e-2,
    continuity_weight: float = 1e-3,
    transition_weight: float = 0.0,
    run_weight: float = 0.0,
    reference_images: Sequence[np.ndarray] | None = None,
    patch_weight: float = 0.0,
    texture_weight: float = 0.0,
    interface_weight: float = 0.0,
    discriminator_lr: float = 1e-4,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_joint_inputs(
        volume,
        vae,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        num_phases=num_phases,
        anchors=anchors,
        anchor_weight=anchor_weight,
        anchor_slab_radius=anchor_slab_radius,
        anchor_slab_weight=anchor_slab_weight,
        entropy_weight=entropy_weight,
        continuity_weight=continuity_weight,
        transition_weight=transition_weight,
        run_weight=run_weight,
        patch_weight=patch_weight,
        texture_weight=texture_weight,
        interface_weight=interface_weight,
        discriminator_lr=discriminator_lr,
    )
    freeze_inference(vae)
    freeze_inference(diffusion_model)

    target_volume, anchor_mask = build_anchor_constraint_volume(
        vae,
        anchors,
        volume_shape=volume.shape,
        num_phases=num_phases,
        segment=anchor_segment,
        device=volume.device,
        dtype=volume.dtype,
    )
    target_fractions = _phase_fraction_target(
        vf_targets,
        num_phases=num_phases,
        device=volume.device,
        dtype=volume.dtype,
    )
    if steps == 0:
        return volume.clone(), {
            "joint_steps": torch.tensor(0, device=volume.device),
        }

    generator = _JointVolumeGenerator(
        volume,
        num_phases=num_phases,
    ).to(device=volume.device, dtype=volume.dtype)
    parameters = list(generator.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    discriminator, discriminator_optimizer, real_images = _build_patch_training(
        reference_images,
        num_phases=num_phases,
        image_size=int(volume.shape[0]),
        device=volume.device,
        dtype=volume.dtype,
        lr=discriminator_lr,
        enabled=patch_weight > 0.0,
    )
    texture_targets = _build_texture_targets(
        reference_images,
        num_phases=num_phases,
        image_size=int(volume.shape[0]),
        device=volume.device,
        dtype=volume.dtype,
        enabled=texture_weight > 0.0,
    )
    interface_target = _build_interface_target(
        reference_images,
        num_phases=num_phases,
        image_size=int(volume.shape[0]),
        device=volume.device,
        dtype=volume.dtype,
        enabled=interface_weight > 0.0,
    )
    transition_target = _build_transition_target(
        reference_images,
        num_phases=num_phases,
        image_size=int(volume.shape[0]),
        device=volume.device,
        dtype=volume.dtype,
        enabled=transition_weight > 0.0,
    )
    run_lengths = tuple(
        length
        for length in _RUN_PROFILE_LENGTHS
        if length <= int(volume.shape[0])
    )
    run_target = _build_run_profile_target(
        reference_images,
        num_phases=num_phases,
        image_size=int(volume.shape[0]),
        device=volume.device,
        dtype=volume.dtype,
        lengths=run_lengths,
        enabled=run_weight > 0.0,
    )
    history: dict[str, list[torch.Tensor]] = {}

    for step in range(steps):
        optimizer.zero_grad()
        logits = generator()
        probabilities = torch.softmax(logits, dim=1)
        axis, indices = select_joint_slices(
            step,
            size=int(volume.shape[0]),
            batch_size=batch_size,
            device=volume.device,
        )
        slice_probabilities = extract_probability_slices(
            probabilities[0],
            axis=axis,
            indices=indices,
        )
        slice_values = _straight_through_phase_values(
            slice_probabilities,
            num_phases=num_phases,
        )

        if discriminator is not None and discriminator_optimizer is not None:
            adversarial_slices = _random_periodic_shift(
                _straight_through_one_hot(
                    _sample_axis_probability_slices(
                        probabilities[0],
                        batch_size=batch_size,
                    )
                )
            )
            discriminator_stats = _update_patch_discriminator(
                discriminator,
                discriminator_optimizer,
                adversarial_slices,
                real_images,
            )
        else:
            adversarial_slices = None
            discriminator_stats = {}

        total = sum(parameter.sum() for parameter in parameters) * 0.0
        stats: dict[str, torch.Tensor] = {}
        stats.update(discriminator_stats)
        if sds_weight > 0.0:
            latent, _ = vae.encode(slice_values)
            prior, _ = sds_loss(
                latent,
                diffusion_model,
                ddpm,
                t_min=t_min,
                t_max=t_max,
            )
            total = total + sds_weight * prior
            stats["sds"] = (sds_weight * prior).detach()

        descriptor_total, descriptor_stats = sample_descriptor_loss(
            slice_values[:, 0],
            num_phases=num_phases,
            vf_targets=vf_targets,
            vf_weight=vf_weight,
            tpc_targets=tpc_targets,
            tpc_weight=tpc_weight,
            sa_targets=sa_targets,
            sa_weight=sa_weight,
            diffusivity_targets=diffusivity_targets,
            diffusivity_solver=diffusivity_solver,
            diffusivity_weight=diffusivity_weight,
            temperature=temperature,
            sa_kernel_size=sa_kernel_size,
            sa_sigma=sa_sigma,
            phase_probabilities=slice_probabilities,
        )
        total = total + descriptor_total
        stats.update(descriptor_stats)

        if target_fractions is not None and vf_weight > 0.0:
            global_vf = probabilities.mean(dim=(0, 2, 3, 4))
            global_vf_loss = 5.0 * vf_weight * F.mse_loss(
                global_vf,
                target_fractions,
            )
            total = total + global_vf_loss
            stats["global_vf"] = global_vf_loss.detach()

        if anchor_weight > 0.0 and bool((anchor_mask > 0).any().item()):
            anchor = _joint_anchor_slab_loss(
                probabilities[0],
                target_volume,
                anchor_mask,
                anchors or [],
                radius=anchor_slab_radius,
                slab_weight=anchor_slab_weight,
            )
            total = total + anchor_weight * anchor
            stats["anchor"] = (anchor_weight * anchor).detach()

        if entropy_weight > 0.0:
            entropy = -(
                probabilities
                * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
            ).sum(dim=1).mean()
            total = total + entropy_weight * entropy
            stats["entropy"] = (entropy_weight * entropy).detach()

        if continuity_weight > 0.0:
            continuity = _continuity_loss(probabilities)
            total = total + continuity_weight * continuity
            stats["continuity"] = (continuity_weight * continuity).detach()

        if transition_weight > 0.0:
            transition, rates = _axis_transition_loss(
                probabilities,
                transition_target,
            )
            total = total + transition_weight * transition
            stats["transition"] = (transition_weight * transition).detach()
            stats["axis_transition_rate"] = rates.detach()

        if run_weight > 0.0:
            categorical = _straight_through_one_hot(probabilities)
            run, run_stats = run_profile_loss(
                categorical,
                run_target,
                lengths=run_lengths,
                weight=run_weight,
            )
            total = total + run
            stats["run_profile"] = run.detach()
            stats["axis_run_profile"] = run_stats[
                "actual_run_profile"
            ].detach()

        if texture_weight > 0.0:
            texture = _texture_swd_loss(slice_probabilities, texture_targets)
            total = total + texture_weight * texture
            stats["texture"] = (texture_weight * texture).detach()

        if interface_weight > 0.0:
            interface = _interface_loss(slice_probabilities, interface_target)
            total = total + interface_weight * interface
            stats["interface"] = (interface_weight * interface).detach()

        if (
            discriminator is not None
            and adversarial_slices is not None
            and patch_weight > 0.0
        ):
            _set_requires_grad(discriminator, False)
            discriminator.eval()
            patch = -discriminator(adversarial_slices).mean()
            ramp_steps = max(1, steps // 5)
            patch_scale = min(1.0, (step + 1) / ramp_steps)
            _set_requires_grad(discriminator, True)
            discriminator.train()
            total = total + patch_weight * patch_scale * patch
            stats["patch"] = (patch_weight * patch_scale * patch).detach()
            stats["patch_scale"] = probabilities.new_tensor(patch_scale)

        stats["loss"] = total.detach()
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        _record(history, stats)

    with torch.no_grad():
        logits = generator()
        probabilities = torch.softmax(logits, dim=1)
        updated = probabilities.argmax(dim=1)[0].to(volume.dtype)

    stats = {
        key: torch.stack(values).mean()
        for key, values in history.items()
        if values
    }
    stats["joint_steps"] = torch.tensor(steps, device=volume.device)
    return updated, stats


class _JointVolumeGenerator(torch.nn.Module):
    def __init__(self, volume: torch.Tensor, *, num_phases: int) -> None:
        super().__init__()
        size = int(volume.shape[0])
        coarse_size = max(2, size // 4)
        probabilities = _initial_volume_probabilities(
            volume,
            num_phases=num_phases,
        )
        coarse = F.interpolate(
            probabilities.unsqueeze(0),
            size=(coarse_size, coarse_size, coarse_size),
            mode="trilinear",
            align_corners=False,
        )
        noise = torch.randn(
            1,
            8,
            coarse_size,
            coarse_size,
            coarse_size,
            device=volume.device,
            dtype=volume.dtype,
        )
        self.register_buffer("condition", torch.cat([coarse, noise], dim=1))
        self.register_buffer(
            "base_logits",
            probabilities.clamp_min(torch.finfo(volume.dtype).tiny)
            .log()
            .unsqueeze(0),
        )
        self.output_size = size
        self.conv16 = _generator_block(num_phases + 8, 48)
        self.conv32 = _generator_block(48, 32)
        self.conv64 = _generator_block(32, 16)
        self.to_logits = torch.nn.Conv3d(16, num_phases, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.to_logits.weight)
        torch.nn.init.zeros_(self.to_logits.bias)

    def forward(self) -> torch.Tensor:
        x = self.conv16(self.condition)
        middle_size = min(self.output_size, int(x.shape[-1]) * 2)
        x = F.interpolate(
            x,
            size=(middle_size, middle_size, middle_size),
            mode="trilinear",
            align_corners=False,
        )
        x = self.conv32(x)
        x = F.interpolate(
            x,
            size=(self.output_size, self.output_size, self.output_size),
            mode="trilinear",
            align_corners=False,
        )
        x = self.conv64(x)
        return self.base_logits + self.to_logits(x)


def _initial_volume_probabilities(
    volume: torch.Tensor,
    *,
    num_phases: int,
) -> torch.Tensor:
    labels = volume.round().clamp(0, num_phases - 1).to(torch.long)
    one_hot = F.one_hot(labels, num_classes=num_phases).permute(3, 0, 1, 2)
    # Preserve the starting labels but leave probability mass for optimization.
    return one_hot.to(volume.dtype) * 0.65 + 0.35 / num_phases


def _generator_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.GroupNorm(8, out_channels),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.GroupNorm(8, out_channels),
        torch.nn.LeakyReLU(0.2, inplace=True),
    )


class _PatchDiscriminator(torch.nn.Module):
    def __init__(self, num_phases: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            _spectral_conv(num_phases, 32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(32, 64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(64, 128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(128, 1, kernel_size=1),
            ),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def _spectral_conv(in_channels: int, out_channels: int) -> torch.nn.Module:
    return torch.nn.utils.spectral_norm(
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
    )


def _reference_one_hot(
    reference_images: Sequence[np.ndarray] | None,
    *,
    num_phases: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not reference_images:
        raise ValueError("reference_images are required for slice texture guidance.")

    arrays = []
    for image in reference_images:
        array = np.asarray(image)
        if array.shape != (image_size, image_size):
            raise ValueError("reference images must match the joint volume slice size.")
        if array.min() < 0 or array.max() >= num_phases:
            raise ValueError("reference images must contain valid phase labels.")
        arrays.append(torch.from_numpy(np.ascontiguousarray(array)).to(torch.long))
    labels = torch.stack(arrays).to(device=device)
    real = F.one_hot(labels, num_classes=num_phases).permute(0, 3, 1, 2)
    return real.to(dtype=dtype)


def _build_patch_training(
    reference_images: Sequence[np.ndarray] | None,
    *,
    num_phases: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    lr: float,
    enabled: bool,
) -> tuple[
    _PatchDiscriminator | None,
    torch.optim.Optimizer | None,
    torch.Tensor | None,
]:
    if not enabled:
        return None, None, None
    real = _reference_one_hot(
        reference_images,
        num_phases=num_phases,
        image_size=image_size,
        device=device,
        dtype=dtype,
    )
    discriminator = _PatchDiscriminator(num_phases).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(0.0, 0.9),
    )
    return discriminator, optimizer, real


def _build_texture_targets(
    reference_images: Sequence[np.ndarray] | None,
    *,
    num_phases: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, int, int]] | None:
    if not enabled:
        return None
    real = _reference_one_hot(
        reference_images,
        num_phases=num_phases,
        image_size=image_size,
        device=device,
        dtype=dtype,
    )
    targets = []
    for pool_size, kernel_size in ((1, 7), (2, 7), (4, 5)):
        patches = _extract_texture_patches(
            real,
            pool_size=pool_size,
            kernel_size=kernel_size,
        )
        if patches.shape[0] > 2048:
            indices = torch.randperm(patches.shape[0], device=device)[:2048]
            patches = patches[indices]
        projection = torch.randn(
            patches.shape[1],
            32,
            device=device,
            dtype=dtype,
        )
        projection = F.normalize(projection, dim=0)
        targets.append((patches.detach(), projection, pool_size, kernel_size))
    return targets


def _texture_swd_loss(
    fake_images: torch.Tensor,
    targets: list[tuple[torch.Tensor, torch.Tensor, int, int]] | None,
) -> torch.Tensor:
    if not targets:
        raise ValueError("texture targets are required when texture guidance is enabled.")
    losses = []
    for target_patches, projection, pool_size, kernel_size in targets:
        fake_patch_batches = _extract_texture_patches(
            fake_images,
            pool_size=pool_size,
            kernel_size=kernel_size,
            flatten=False,
        )
        for fake_patches in fake_patch_batches:
            sample_count = min(
                512,
                int(fake_patches.shape[0]),
                int(target_patches.shape[0]),
            )
            fake_indices = torch.randperm(
                fake_patches.shape[0],
                device=fake_patches.device,
            )[:sample_count]
            target_indices = torch.randperm(
                target_patches.shape[0],
                device=target_patches.device,
            )[:sample_count]
            fake_projection = (
                fake_patches[fake_indices] @ projection
            ).sort(dim=0).values
            target_projection = (
                target_patches[target_indices] @ projection
            ).sort(dim=0).values
            losses.append((fake_projection - target_projection).abs().mean())
    return torch.stack(losses).mean()


def _extract_texture_patches(
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
    )
    patches = patches.transpose(1, 2)
    if flatten:
        return patches.reshape(-1, patches.shape[-1])
    return patches


def _build_interface_target(
    reference_images: Sequence[np.ndarray] | None,
    *,
    num_phases: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> torch.Tensor | None:
    if not enabled:
        return None
    real = _reference_one_hot(
        reference_images,
        num_phases=num_phases,
        image_size=image_size,
        device=device,
        dtype=dtype,
    )
    return _phase_interface_matrices(real).mean(dim=0).detach()


def _build_transition_target(
    reference_images: Sequence[np.ndarray] | None,
    *,
    num_phases: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool,
) -> torch.Tensor | None:
    if not enabled:
        return None
    real = _reference_one_hot(
        reference_images,
        num_phases=num_phases,
        image_size=image_size,
        device=device,
        dtype=dtype,
    )
    horizontal = 1.0 - (real[:, :, :, :-1] * real[:, :, :, 1:]).sum(dim=1)
    vertical = 1.0 - (real[:, :, :-1, :] * real[:, :, 1:, :]).sum(dim=1)
    return 0.5 * (horizontal.mean() + vertical.mean())


def _build_run_profile_target(
    reference_images: Sequence[np.ndarray] | None,
    *,
    num_phases: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    lengths: Sequence[int],
    enabled: bool,
) -> torch.Tensor | None:
    if not enabled:
        return None
    if not lengths:
        raise ValueError("run profile requires a volume size of at least 2.")
    real = _reference_one_hot(
        reference_images,
        num_phases=num_phases,
        image_size=image_size,
        device=device,
        dtype=dtype,
    )
    return compute_run_profile(real, lengths=lengths).mean(dim=0).detach()


def _axis_transition_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target is None:
        raise ValueError(
            "transition target is required when transition guidance is enabled."
        )
    categorical = _straight_through_one_hot(probabilities)
    rates = []
    for dimension in (2, 3, 4):
        length = int(categorical.shape[dimension])
        before = categorical.narrow(dimension, 0, length - 1)
        after = categorical.narrow(dimension, 1, length - 1)
        rates.append(1.0 - (before * after).sum(dim=1).mean())
    axis_rates = torch.stack(rates)
    return F.mse_loss(axis_rates, target.expand_as(axis_rates)), axis_rates


def _interface_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor | None,
) -> torch.Tensor:
    if target is None:
        raise ValueError("interface target is required when interface guidance is enabled.")
    actual = _phase_interface_matrices(probabilities)
    expected = target.unsqueeze(0).expand_as(actual)
    num_phases = int(actual.shape[1])
    pair_loss = F.mse_loss(actual, expected) * num_phases**2
    transition_loss = F.mse_loss(
        actual.sum(dim=(1, 2)),
        expected.sum(dim=(1, 2)),
    )
    return pair_loss + transition_loss


def _phase_interface_matrices(probabilities: torch.Tensor) -> torch.Tensor:
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
        horizontal
        + horizontal.transpose(1, 2)
        + vertical
        + vertical.transpose(1, 2)
    )
    off_diagonal = 1.0 - torch.eye(
        num_phases,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return symmetric * off_diagonal.unsqueeze(0)


def _update_patch_discriminator(
    discriminator: _PatchDiscriminator,
    optimizer: torch.optim.Optimizer,
    fake_probabilities: torch.Tensor,
    real_images: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if real_images is None:
        return {}
    _set_requires_grad(discriminator, True)
    discriminator.train()
    optimizer.zero_grad()
    batch_size = min(8, int(fake_probabilities.shape[0]), int(real_images.shape[0]))
    real_indices = torch.randint(
        0,
        real_images.shape[0],
        (batch_size,),
        device=real_images.device,
    )
    real = _random_periodic_shift(real_images[real_indices])
    fake = fake_probabilities.detach()
    fake_indices = torch.randperm(fake.shape[0], device=fake.device)[:batch_size]
    real_score = discriminator(real)
    fake_score = discriminator(fake[fake_indices])
    loss = (
        F.relu(1.0 - real_score).mean()
        + F.relu(1.0 + fake_score).mean()
    )
    loss.backward()
    optimizer.step()
    return {
        "patch_discriminator": loss.detach(),
        "patch_real": real_score.mean().detach(),
        "patch_fake": fake_score.mean().detach(),
        "patch_margin": (real_score.mean() - fake_score.mean()).detach(),
    }


def _random_periodic_shift(images: torch.Tensor) -> torch.Tensor:
    height, width = images.shape[-2:]
    shifts = torch.randint(
        0,
        min(height, width),
        (images.shape[0], 2),
        device=images.device,
    )
    return torch.stack(
        [
            torch.roll(image, tuple(map(int, shift)), dims=(-2, -1))
            for image, shift in zip(images, shifts)
        ]
    )


def _straight_through_one_hot(probabilities: torch.Tensor) -> torch.Tensor:
    hard_indices = probabilities.argmax(dim=1)
    hard = F.one_hot(
        hard_indices,
        num_classes=int(probabilities.shape[1]),
    ).movedim(-1, 1)
    return hard.to(probabilities.dtype) + probabilities - probabilities.detach()


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def select_joint_slices(
    step: int,
    *,
    size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[int, list[int]]:
    cycle = step // 3
    axis = _AXIS_ORDERS[cycle % len(_AXIS_ORDERS)][step % 3]
    indices = torch.randperm(size, device=device)[:batch_size].tolist()
    return axis, [int(index) for index in indices]


def extract_probability_slices(
    probabilities: torch.Tensor,
    *,
    axis: int,
    indices: Sequence[int],
) -> torch.Tensor:
    index = torch.as_tensor(indices, device=probabilities.device, dtype=torch.long)
    if axis == 0:
        return probabilities[:, index, :, :].permute(1, 0, 2, 3)
    if axis == 1:
        return probabilities[:, :, index, :].permute(2, 0, 1, 3)
    return probabilities[:, :, :, index].permute(3, 0, 1, 2)


def all_axis_probability_slices(probabilities: torch.Tensor) -> torch.Tensor:
    size = int(probabilities.shape[1])
    indices = list(range(size))
    return torch.cat(
        [
            extract_probability_slices(
                probabilities,
                axis=axis,
                indices=indices,
            )
            for axis in range(3)
        ],
        dim=0,
    )


def _sample_axis_probability_slices(
    probabilities: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Sample a balanced categorical-critic batch from all three orientations."""
    size = int(probabilities.shape[1])
    base_count, remainder = divmod(batch_size, 3)
    batches = []
    for axis in range(3):
        count = base_count + int(axis < remainder)
        if count == 0:
            continue
        indices = torch.randperm(size, device=probabilities.device)[:count].tolist()
        batches.append(
            extract_probability_slices(
                probabilities,
                axis=axis,
                indices=indices,
            )
        )
    return torch.cat(batches, dim=0)


def _straight_through_phase_values(
    probabilities: torch.Tensor,
    *,
    num_phases: int,
) -> torch.Tensor:
    hard_indices = probabilities.argmax(dim=1)
    hard = F.one_hot(hard_indices, num_classes=num_phases).permute(0, 3, 1, 2)
    straight_through = hard.to(probabilities.dtype) + probabilities - probabilities.detach()
    levels = phase_levels(
        num_phases,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return (straight_through * levels.view(1, num_phases, 1, 1)).sum(
        dim=1,
        keepdim=True,
    )


def _joint_anchor_slab_loss(
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
        anchor_target = _volume_plane(target, int(anchor.axis), int(anchor.index))
        for offset in range(-radius, radius + 1):
            index = int(anchor.index) + offset
            if index < 0 or index >= size:
                continue
            weight = (
                1.0
                if offset == 0
                else slab_weight * (radius + 1 - abs(offset)) / (radius + 1)
            )
            if weight == 0.0:
                continue
            plane = _probability_plane(probabilities, int(anchor.axis), index)
            losses.append(
                F.nll_loss(
                    plane.clamp_min(torch.finfo(plane.dtype).tiny)
                    .log()
                    .unsqueeze(0),
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


def _volume_plane(values: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    if axis == 0:
        return values[index, :, :]
    if axis == 1:
        return values[:, index, :]
    return values[:, :, index]


def _probability_plane(values: torch.Tensor, axis: int, index: int) -> torch.Tensor:
    if axis == 0:
        return values[:, index, :, :]
    if axis == 1:
        return values[:, :, index, :]
    return values[:, :, :, index]


def _continuity_loss(probabilities: torch.Tensor) -> torch.Tensor:
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


def _phase_fraction_target(
    targets: Mapping[int, float] | torch.Tensor | None,
    *,
    num_phases: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if targets is None:
        return None
    return build_phase_target(
        targets,
        num_phases=num_phases,
        device=device,
        dtype=dtype,
        label="fraction",
        require_sum_one=True,
    )


def _record(
    history: dict[str, list[torch.Tensor]],
    stats: dict[str, torch.Tensor],
) -> None:
    for key, value in stats.items():
        history.setdefault(key, []).append(value.detach())


def _validate_joint_inputs(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    num_phases: int,
    anchors: Sequence[AnchorSlice] | None,
    anchor_weight: float,
    anchor_slab_radius: int,
    anchor_slab_weight: float,
    entropy_weight: float,
    continuity_weight: float,
    transition_weight: float,
    run_weight: float,
    patch_weight: float,
    texture_weight: float,
    interface_weight: float,
    discriminator_lr: float,
) -> None:
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError("joint 3D optimization requires a cubic [D, H, W] volume.")
    if int(volume.shape[0]) != int(vae.image_size):
        raise ValueError("joint 3D volume size must match vae.image_size.")
    if getattr(vae, "num_phases", None) != num_phases:
        raise ValueError("joint 3D optimization requires a matching categorical VAE.")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError("batch_size must be an integer.")
    if batch_size <= 0 or batch_size > int(volume.shape[0]):
        raise ValueError("batch_size must be between 1 and the volume size.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    for name, weight in (
        ("anchor_weight", anchor_weight),
        ("entropy_weight", entropy_weight),
        ("continuity_weight", continuity_weight),
        ("transition_weight", transition_weight),
        ("run_weight", run_weight),
        ("patch_weight", patch_weight),
        ("texture_weight", texture_weight),
        ("interface_weight", interface_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if not isinstance(anchor_slab_radius, int) or isinstance(anchor_slab_radius, bool):
        raise ValueError("anchor_slab_radius must be an integer.")
    if anchor_slab_radius < 0:
        raise ValueError("anchor_slab_radius must be non-negative.")
    if anchor_slab_weight < 0.0 or anchor_slab_weight > 1.0:
        raise ValueError("anchor_slab_weight must be between 0 and 1.")
    if discriminator_lr <= 0.0:
        raise ValueError("discriminator_lr must be positive.")
    if anchor_weight > 0.0 and not anchors:
        raise ValueError("anchors are required when anchor_weight is positive.")
