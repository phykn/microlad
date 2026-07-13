from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.modeling.diffusion import DDPMProcess
from src.modeling.inference import freeze
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.guidance.metrics.runs import (
    run_profile_loss,
)
from src.pipelines.guidance.metrics.loss import sample_descriptor_loss
from src.pipelines.guidance.metrics.conductance import ConductanceSolver
from src.pipelines.guidance.conditioning.prepare import (
    build_anchor_constraint_volume,
)
from src.pipelines.guidance.sds.prior import sds_loss
from src.pipelines.guidance.metrics.targets import build_phase_target
from src.pipelines.guidance.joint.loss import (
    anchor_loss,
    axis_transition_loss,
    continuity_loss,
    interface_loss,
    texture_loss,
    update_discriminator,
)
from src.pipelines.guidance.joint.model import JointGenerator, build_patch_training
from src.pipelines.guidance.joint.slices import (
    extract_slices,
    periodic_shift,
    phase_values,
    sample_slices,
    select_slices,
    straight_through_one_hot,
)
from src.pipelines.guidance.joint.targets import (
    interface_target as build_interface_target,
    reference_one_hot,
    run_target as build_run_target,
    texture_targets as build_texture_targets,
    transition_target as build_transition_target,
)
from src.pipelines.guidance.joint.validation import validate_inputs


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
    segment_anchors: bool = False,
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
    diffusivity_solver: ConductanceSolver | None = None,
    diffusivity_weight: float = 0.0,
    entropy_weight: float = 1e-2,
    continuity_weight: float = 1e-3,
    transition_weight: float = 0.0,
    run_weight: float = 0.0,
    reference_labels: torch.Tensor | None = None,
    patch_weight: float = 0.0,
    texture_weight: float = 0.0,
    interface_weight: float = 0.0,
    discriminator_lr: float = 1e-4,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    validate_inputs(
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
    freeze(vae)
    freeze(diffusion_model)

    target_volume, anchor_mask = build_anchor_constraint_volume(
        vae,
        anchors,
        volume_shape=volume.shape,
        num_phases=num_phases,
        segment=segment_anchors,
        device=volume.device,
        dtype=volume.dtype,
    )
    target_fractions = (
        None
        if vf_targets is None
        else build_phase_target(
            vf_targets,
            num_phases=num_phases,
            device=volume.device,
            dtype=volume.dtype,
            label="fraction",
            require_sum_one=True,
        )
    )
    if steps == 0:
        return volume.clone(), {
            "joint_steps": torch.tensor(0, device=volume.device),
        }

    generator = JointGenerator(
        volume,
        num_phases=num_phases,
    ).to(device=volume.device, dtype=volume.dtype)
    parameters = list(generator.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    uses_references = any(
        weight > 0.0
        for weight in (
            transition_weight,
            run_weight,
            patch_weight,
            texture_weight,
            interface_weight,
        )
    )
    references = (
        reference_one_hot(
            reference_labels,
            num_phases=num_phases,
            image_size=int(volume.shape[0]),
            device=volume.device,
            dtype=volume.dtype,
        )
        if uses_references
        else None
    )
    discriminator, discriminator_optimizer, real_images = build_patch_training(
        references,
        num_phases=num_phases,
        device=volume.device,
        dtype=volume.dtype,
        lr=discriminator_lr,
        enabled=patch_weight > 0.0,
    )
    texture_targets = build_texture_targets(
        references,
        device=volume.device,
        dtype=volume.dtype,
        enabled=texture_weight > 0.0,
    )
    interface_target = build_interface_target(
        references,
        enabled=interface_weight > 0.0,
    )
    transition_target = build_transition_target(
        references,
        enabled=transition_weight > 0.0,
    )
    run_lengths = tuple(
        length for length in _RUN_PROFILE_LENGTHS if length <= int(volume.shape[0])
    )
    run_target = build_run_target(
        references,
        lengths=run_lengths,
        enabled=run_weight > 0.0,
    )
    history: dict[str, list[torch.Tensor]] = {}

    for step in range(steps):
        optimizer.zero_grad()
        logits = generator()
        probabilities = torch.softmax(logits, dim=1)
        axis, indices = select_slices(
            step,
            size=int(volume.shape[0]),
            batch_size=batch_size,
            device=volume.device,
        )
        slice_probabilities = extract_slices(
            probabilities[0],
            axis=axis,
            indices=indices,
        )
        slice_values = phase_values(
            slice_probabilities,
            num_phases=num_phases,
        )

        if discriminator is not None and discriminator_optimizer is not None:
            adversarial_slices = periodic_shift(
                straight_through_one_hot(
                    sample_slices(
                        probabilities[0],
                        batch_size=batch_size,
                    )
                )
            )
            discriminator_stats = update_discriminator(
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
            global_vf_loss = (
                5.0
                * vf_weight
                * F.mse_loss(
                    global_vf,
                    target_fractions,
                )
            )
            total = total + global_vf_loss
            stats["global_vf"] = global_vf_loss.detach()

        if anchor_weight > 0.0 and bool((anchor_mask > 0).any().item()):
            anchor = anchor_loss(
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
            entropy = (
                -(
                    probabilities
                    * probabilities.clamp_min(
                        torch.finfo(probabilities.dtype).tiny
                    ).log()
                )
                .sum(dim=1)
                .mean()
            )
            total = total + entropy_weight * entropy
            stats["entropy"] = (entropy_weight * entropy).detach()

        if continuity_weight > 0.0:
            continuity = continuity_loss(probabilities)
            total = total + continuity_weight * continuity
            stats["continuity"] = (continuity_weight * continuity).detach()

        if transition_weight > 0.0:
            transition, rates = axis_transition_loss(
                probabilities,
                transition_target,
            )
            total = total + transition_weight * transition
            stats["transition"] = (transition_weight * transition).detach()
            stats["axis_transition_rate"] = rates.detach()

        if run_weight > 0.0:
            categorical = straight_through_one_hot(probabilities)
            run, run_stats = run_profile_loss(
                categorical,
                run_target,
                lengths=run_lengths,
                weight=run_weight,
            )
            total = total + run
            stats["run_profile"] = run.detach()
            stats["axis_run_profile"] = run_stats["actual_run_profile"].detach()

        if texture_weight > 0.0:
            texture = texture_loss(slice_probabilities, texture_targets)
            total = total + texture_weight * texture
            stats["texture"] = (texture_weight * texture).detach()

        if interface_weight > 0.0:
            interface = interface_loss(slice_probabilities, interface_target)
            total = total + interface_weight * interface
            stats["interface"] = (interface_weight * interface).detach()

        if (
            discriminator is not None
            and adversarial_slices is not None
            and patch_weight > 0.0
        ):
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            discriminator.eval()
            patch = -discriminator(adversarial_slices).mean()
            ramp_steps = max(1, steps // 5)
            patch_scale = min(1.0, (step + 1) / ramp_steps)
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            discriminator.train()
            total = total + patch_weight * patch_scale * patch
            stats["patch"] = (patch_weight * patch_scale * patch).detach()
            stats["patch_scale"] = probabilities.new_tensor(patch_scale)

        stats["loss"] = total.detach()
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        for key, value in stats.items():
            history.setdefault(key, []).append(value.detach())

    with torch.no_grad():
        logits = generator()
        probabilities = torch.softmax(logits, dim=1)
        updated = probabilities.argmax(dim=1)[0].to(volume.dtype)

    stats = {
        f"history_{key}": torch.stack(values).mean(dim=0)
        for key, values in history.items()
        if values
    }
    stats["joint_steps"] = torch.tensor(steps, device=volume.device)
    return updated, stats
