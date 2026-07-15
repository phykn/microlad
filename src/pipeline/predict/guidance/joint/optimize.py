from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.modeling.gan import (
    ImageCritic,
    guidance_loss,
    morphology_feature_loss,
)
from src.pipeline.predict.guidance.latent_slices import sample_slices
from src.modeling.diffusion import DDPMProcess
from src.modeling.inference import freeze
from src.modeling.phases import geometric_probability_consensus
from src.pipeline.predict.guidance.conditioning.model import AnchorSlice
from src.pipeline.predict.guidance.conditioning.prepare import build_anchor_constraint_volume
from src.pipeline.predict.guidance.joint.loss import (
    anchor_loss,
    axis_loss,
    axis_mass_loss,
    continuity_loss,
    fraction_loss,
)
from src.pipeline.predict.guidance.joint.model import LatentRefiner
from src.pipeline.predict.guidance.joint.slices import (
    extract_slices,
    phase_values,
    select_slices,
    straight_through_one_hot,
)
from src.pipeline.predict.guidance.metrics.conductance import ConductanceSolver
from src.pipeline.predict.guidance.metrics.loss import sample_descriptor_loss
from src.pipeline.predict.guidance.metrics.targets import build_phase_target
from src.pipeline.predict.guidance.prior import sds_loss
from src.pipeline.predict.reconstruction.volume import decode_axis_probs


def optimize_latent(
    latent: torch.Tensor,
    vae: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    ddpm: DDPMProcess,
    *,
    steps: int,
    batch_size: int,
    decode_batch_size: int | None = 2,
    lr: float,
    t_min: int,
    t_max: int,
    num_phases: int,
    anchors: Sequence[AnchorSlice] | None = None,
    segment_anchors: bool = False,
    sds_weight: float = 1.0,
    critic: ImageCritic | None = None,
    critic_weight: float = 0.0,
    critic_mode: str = "score",
    critic_references: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
    fraction_targets: Mapping[int, float] | torch.Tensor | None = None,
    phase_fractions: torch.Tensor | None = None,
    slice_fraction_weight: float = 0.0,
    global_fraction_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: ConductanceSolver | None = None,
    diffusivity_weight: float = 0.0,
    axis_weight: float = 1.0,
    axis_mass_weight: float = 0.25,
    continuity_weight: float = 1e-3,
    residual_scale: float = 0.25,
    preservation_weight: float = 5.0,
    temperature: float = 0.1,
    sa_kernel_size: int = 7,
    sa_sigma: float = 1.0,
    progress: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_inputs(
        latent,
        vae,
        steps=steps,
        batch_size=batch_size,
        decode_batch_size=decode_batch_size,
        lr=lr,
        num_phases=num_phases,
        critic=critic,
        critic_weight=critic_weight,
        critic_mode=critic_mode,
        critic_references=critic_references,
        progress=progress,
    )
    freeze(vae)
    freeze(diffusion_model)
    if critic is not None:
        freeze(critic)

    if steps == 0:
        return latent.detach().clone(), {
            "step": torch.empty(0, device=latent.device, dtype=torch.long)
        }

    output_size = int(vae.image_size)
    target_volume, anchor_voxels = build_anchor_constraint_volume(
        vae,
        anchors,
        volume_shape=(output_size, output_size, output_size),
        num_phases=num_phases,
        segment=segment_anchors,
        device=latent.device,
        dtype=latent.dtype,
    )
    target_fractions = (
        None
        if fraction_targets is None
        else build_phase_target(
            fraction_targets,
            num_phases=num_phases,
            device=latent.device,
            dtype=latent.dtype,
            label="fraction",
            require_sum_one=True,
        )
    )
    base = latent.detach().unsqueeze(0)
    refiner = LatentRefiner(
        int(latent.shape[0]),
        scale=residual_scale,
    ).to(device=latent.device, dtype=latent.dtype)
    parameters = list(refiner.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history: dict[str, list[torch.Tensor]] = {}

    step_range = tqdm(
        range(steps),
        total=steps,
        desc="Joint guidance",
        disable=not progress,
    )
    for step in step_range:
        optimizer.zero_grad(set_to_none=True)
        refined = refiner(base)
        latent_slices = sample_slices(
            refined,
            count=batch_size,
            crop_size=int(vae.latent_size),
            axis_offset=step % 3,
        )
        total = refined.sum() * 0.0
        stats: dict[str, torch.Tensor] = {}
        display: dict[str, torch.Tensor] = {}

        if sds_weight > 0.0:
            prior, _ = sds_loss(
                latent_slices,
                diffusion_model,
                ddpm,
                t_min=t_min,
                t_max=t_max,
                phase_fractions=phase_fractions,
            )
            total = total + sds_weight * prior
            stats["sds"] = (sds_weight * prior).detach()

        axis_probabilities = decode_axis_probs(
            vae,
            refined[0],
            num_phases=num_phases,
            plane_batch_size=decode_batch_size,
            checkpoint_gradients=decode_batch_size is not None,
        )
        probabilities = geometric_probability_consensus(
            axis_probabilities,
            num_phases,
        ).unsqueeze(0)
        axis, indices = select_slices(
            step,
            size=output_size,
            batch_size=batch_size,
            device=latent.device,
        )
        slice_probabilities = extract_slices(
            probabilities[0],
            axis=axis,
            indices=indices,
        )
        slice_values = phase_values(slice_probabilities, num_phases=num_phases)

        if critic is not None and critic_weight > 0.0:
            critic_probabilities = straight_through_one_hot(slice_probabilities)
            critic_term = (
                guidance_loss(critic(critic_probabilities))
                if critic_mode == "score"
                else morphology_feature_loss(
                    critic,
                    critic_probabilities,
                    critic_references,
                )
            )
            total = total + critic_weight * critic_term
            stats["critic"] = (critic_weight * critic_term).detach()
            display["critic"] = critic_term.detach()

        descriptor_total, descriptor_stats = sample_descriptor_loss(
            slice_values[:, 0],
            num_phases=num_phases,
            fraction_targets=fraction_targets,
            fraction_weight=slice_fraction_weight,
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

        if target_fractions is not None and global_fraction_weight > 0.0:
            hard_probabilities = (
                F.one_hot(
                    probabilities.argmax(dim=1),
                    num_classes=num_phases,
                )
                .movedim(-1, 1)
                .to(probabilities.dtype)
            )
            categorical_probabilities = (
                hard_probabilities + probabilities - probabilities.detach()
            )
            global_fraction = categorical_probabilities.mean(dim=(0, 2, 3, 4))
            fraction_error = fraction_loss(
                global_fraction,
                target_fractions,
            )
            fraction = global_fraction_weight * fraction_error
            total = total + fraction
            stats["global_fraction"] = fraction.detach()
            display["global_fraction"] = fraction_error.detach()

        if anchor_weight > 0.0 and bool((anchor_voxels > 0).any().item()):
            anchor = anchor_loss(
                probabilities[0],
                target_volume,
                anchor_voxels,
            )
            total = total + anchor_weight * anchor
            stats["anchor"] = (anchor_weight * anchor).detach()
            display["anchor"] = anchor.detach()

        if axis_weight > 0.0:
            axis = axis_loss(axis_probabilities)
            total = total + axis_weight * axis
            stats["axis"] = (axis_weight * axis).detach()
            display["axis"] = axis.detach()

        if axis_mass_weight > 0.0:
            axis_mass = axis_mass_loss(axis_probabilities)
            total = total + axis_mass_weight * axis_mass
            stats["axis_mass"] = (axis_mass_weight * axis_mass).detach()

        if continuity_weight > 0.0:
            continuity = continuity_loss(probabilities)
            total = total + continuity_weight * continuity
            stats["continuity"] = (continuity_weight * continuity).detach()

        delta = refined - base
        preservation = delta.square().mean()
        total = total + preservation_weight * preservation
        stats["preservation"] = (preservation_weight * preservation).detach()
        stats["loss"] = total.detach()

        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        for key, value in stats.items():
            history.setdefault(key, []).append(value.detach())

        completed = step + 1
        refresh_every = max(1, steps // 100)
        if progress and (
            completed == 1 or completed % refresh_every == 0 or completed == steps
        ):
            postfix = {"loss": f"{float(stats['loss'].item()):.4g}"}
            for name, label in (
                ("anchor", "anchor"),
                ("critic", "critic"),
                ("global_fraction", "fraction"),
                ("axis", "axis"),
            ):
                if name in display:
                    postfix[label] = f"{float(display[name].item()):.4g}"
            step_range.set_postfix(postfix)
    with torch.no_grad():
        final_latent = refiner(base)[0].detach().clone()
    history_tensors = {
        key: torch.stack(values)
        for key, values in history.items()
        if values
    }
    history_tensors["step"] = torch.arange(
        1,
        steps + 1,
        device=latent.device,
    )
    return final_latent, history_tensors


def _validate_inputs(
    latent: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
    batch_size: int,
    decode_batch_size: int | None,
    lr: float,
    num_phases: int,
    critic: ImageCritic | None,
    critic_weight: float,
    critic_mode: str,
    critic_references: torch.Tensor | None,
    progress: bool,
) -> None:
    expected = (
        int(vae.latent_ch),
        int(vae.latent_size),
        int(vae.latent_size),
        int(vae.latent_size),
    )
    if latent.shape != expected:
        raise ValueError(f"joint latent must have shape {expected}.")
    if not latent.is_floating_point() or not torch.isfinite(latent).all():
        raise ValueError("joint latent must contain finite floating-point values.")
    if (
        steps < 0
        or batch_size <= 0
        or (decode_batch_size is not None and decode_batch_size <= 0)
        or lr <= 0.0
    ):
        raise ValueError("joint optimization settings are invalid.")
    if num_phases != getattr(vae, "num_phases", None):
        raise ValueError("num_phases must match vae.num_phases.")
    if not isinstance(progress, bool):
        raise ValueError("progress must be a boolean.")
    if critic_weight > 0.0 and critic is None:
        raise ValueError("critic is required when critic_weight is positive.")
    if critic_mode not in ("score", "feature"):
        raise ValueError("critic_mode must be 'score' or 'feature'.")
    if critic_weight > 0.0 and critic_mode == "feature":
        if critic_references is None:
            raise ValueError(
                "critic_references are required for feature critic guidance."
            )
        if (
            critic_references.ndim != 4
            or critic_references.shape[0] <= 0
            or critic_references.shape[1] != num_phases
            or not critic_references.is_floating_point()
            or not torch.isfinite(critic_references).all()
            or critic_references.device != latent.device
        ):
            raise ValueError(
                "critic_references must contain finite floating-point "
                "[N, num_phases, H, W] probabilities."
            )
