from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.modeling.critic import sample_slices
from src.modeling.diffusion import DDPMProcess
from src.modeling.inference import freeze
from src.pipelines.guidance.conditioning.images import prepare_anchor_image
from src.pipelines.guidance.conditioning.model import AnchorSlice
from src.pipelines.guidance.joint.loss import fraction_loss
from src.pipelines.guidance.metrics.conductance import ConductanceSolver
from src.pipelines.guidance.metrics.loss import sample_descriptor_loss
from src.pipelines.guidance.metrics.targets import build_phase_target
from src.pipelines.guidance.prior import sds_loss
from src.pipelines.reconstruction.volume import decode_latents
from src.pipelines.scaling.decoding import decode_anchor_patch
from src.validation import require_finite, require_finite_number, require_int


def optimize_large_latent(
    latent: torch.Tensor,
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
    fraction_targets: Mapping[int, float] | torch.Tensor | None = None,
    slice_fraction_weight: float = 0.0,
    global_fraction_weight: float = 0.0,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    tpc_weight: float = 0.0,
    sa_targets: Mapping[int, float] | torch.Tensor | None = None,
    sa_weight: float = 0.0,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None = None,
    diffusivity_solver: ConductanceSolver | None = None,
    diffusivity_weight: float = 0.0,
    continuity_weight: float = 0.05,
    preservation_weight: float = 1.0,
    residual_scale: float = 1.0,
    checkpoint_every: int = 100,
    decode_batch_size: int | None = 16,
    tile_overlap: int = 0,
    temperature: float = 0.1,
    progress: bool = False,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    _validate(
        latent,
        vae,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        num_phases=num_phases,
        anchors=anchors,
        segment_anchors=segment_anchors,
        anchor_weight=anchor_weight,
        fraction_targets=fraction_targets,
        slice_fraction_weight=slice_fraction_weight,
        global_fraction_weight=global_fraction_weight,
        tpc_targets=tpc_targets,
        tpc_weight=tpc_weight,
        sa_targets=sa_targets,
        sa_weight=sa_weight,
        diffusivity_targets=diffusivity_targets,
        diffusivity_solver=diffusivity_solver,
        diffusivity_weight=diffusivity_weight,
        sds_weight=sds_weight,
        continuity_weight=continuity_weight,
        preservation_weight=preservation_weight,
        residual_scale=residual_scale,
        checkpoint_every=checkpoint_every,
        decode_batch_size=decode_batch_size,
        temperature=temperature,
        progress=progress,
    )
    freeze(vae)
    freeze(diffusion_model)

    candidates = [latent.detach().clone()]
    candidate_steps = [0]
    if steps == 0:
        return tuple(candidates), {
            "scale_steps": torch.tensor(0, device=latent.device),
            "scale_candidate_steps": torch.tensor(
                candidate_steps,
                device=latent.device,
            ),
        }

    target_fraction = (
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
    prepared_anchors = [
        (
            anchor,
            prepare_anchor_image(
                anchor.image,
                num_phases=num_phases,
                segment=segment_anchors,
            )[0, 0].to(device=latent.device),
        )
        for anchor in anchors or ()
    ]
    base = latent.detach().unsqueeze(0)
    scale = base.std(
        dim=(2, 3, 4),
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-3)
    residual = torch.zeros_like(base, requires_grad=True)
    optimizer = torch.optim.Adam([residual], lr=lr)
    history: dict[str, list[torch.Tensor]] = {}

    step_range = tqdm(
        range(steps),
        total=steps,
        desc="Scale guidance",
        disable=not progress,
    )
    for step in step_range:
        optimizer.zero_grad(set_to_none=True)
        refined = base + residual_scale * scale * torch.tanh(residual)
        crops = sample_slices(
            refined,
            count=batch_size,
            crop_size=int(vae.latent_size),
            axis_offset=step % 3,
        )
        values, probabilities = decode_latents(
            vae,
            crops,
            num_phases=num_phases,
        )
        total = refined.sum() * 0.0
        stats: dict[str, torch.Tensor] = {}

        if sds_weight > 0.0:
            prior, _ = sds_loss(
                crops,
                diffusion_model,
                ddpm,
                t_min=t_min,
                t_max=t_max,
            )
            total = total + sds_weight * prior
            stats["sds"] = (sds_weight * prior).detach()

        descriptor, descriptor_stats = sample_descriptor_loss(
            values,
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
            phase_probabilities=probabilities,
        )
        total = total + descriptor
        stats.update(descriptor_stats)

        if target_fraction is not None and global_fraction_weight > 0.0:
            hard = (
                F.one_hot(
                    probabilities.argmax(dim=1),
                    num_classes=num_phases,
                )
                .movedim(-1, 1)
                .to(probabilities.dtype)
            )
            categorical = hard + probabilities - probabilities.detach()
            measured = categorical.mean(dim=(0, 2, 3))
            error = fraction_loss(measured, target_fraction)
            total = total + global_fraction_weight * error
            stats["global_fraction"] = (
                global_fraction_weight * error
            ).detach()

        if anchor_weight > 0.0:
            anchor = _decoded_anchor_loss(
                refined[0],
                vae,
                prepared_anchors,
                step=step,
                num_phases=num_phases,
                tile_overlap=tile_overlap,
                decode_batch_size=decode_batch_size,
            )
            total = total + anchor_weight * anchor
            stats["anchor"] = (anchor_weight * anchor).detach()

        if continuity_weight > 0.0:
            continuity = _continuity_loss(refined, base)
            total = total + continuity_weight * continuity
            stats["continuity"] = (continuity_weight * continuity).detach()

        preservation = (refined - base).square().mean()
        total = total + preservation_weight * preservation
        stats["preservation"] = (preservation_weight * preservation).detach()
        stats["loss"] = total.detach()

        total.backward()
        torch.nn.utils.clip_grad_norm_([residual], max_norm=1.0)
        optimizer.step()
        for name, value in stats.items():
            history.setdefault(name, []).append(value.detach())

        completed = step + 1
        refresh_every = max(1, steps // 100)
        if progress and (
            completed == 1 or completed % refresh_every == 0 or completed == steps
        ):
            display = {"loss": f"{float(stats['loss'].item()):.4g}"}
            for name in ("anchor", "global_fraction", "sds"):
                if name in stats:
                    display[name] = f"{float(stats[name].item()):.4g}"
            step_range.set_postfix(display)

        if completed % checkpoint_every == 0 or completed == steps:
            with torch.no_grad():
                candidate = base + residual_scale * scale * torch.tanh(residual)
                candidates.append(candidate[0].detach().clone())
            candidate_steps.append(completed)

    summary = {
        f"history_{name}": torch.stack(values).mean(dim=0)
        for name, values in history.items()
        if values
    }
    summary.update(
        {
            f"scale_final_{name}": values[-1]
            for name, values in history.items()
            if values
        }
    )
    summary["scale_steps"] = torch.tensor(steps, device=latent.device)
    summary["scale_candidate_steps"] = torch.tensor(
        candidate_steps,
        device=latent.device,
    )
    deltas = torch.stack([candidate - latent for candidate in candidates])
    summary["scale_candidate_delta_rms"] = deltas.square().mean(
        dim=(1, 2, 3, 4)
    ).sqrt()
    summary["scale_candidate_delta_max"] = deltas.abs().amax(
        dim=(1, 2, 3, 4)
    )
    summary["scale_base_std"] = latent.std()
    return tuple(candidates), summary


def _continuity_loss(refined: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            F.mse_loss(torch.diff(refined, dim=axis), torch.diff(base, dim=axis))
            for axis in (2, 3, 4)
        ]
    ).mean()


def _decoded_anchor_loss(
    latent: torch.Tensor,
    vae: torch.nn.Module,
    anchors: Sequence[tuple[AnchorSlice, torch.Tensor]],
    *,
    step: int,
    num_phases: int,
    tile_overlap: int,
    decode_batch_size: int | None,
) -> torch.Tensor:
    anchor, target = anchors[step % len(anchors)]
    target_size = int(target.shape[0])
    crop_size = min(max(int(vae.image_size) // 2, 1), target_size)
    last_start = target_size - crop_size
    starts = tuple(
        dict.fromkeys(
            min(start, last_start) for start in range(0, target_size, crop_size)
        )
    )
    crop_grid = tuple((row, col) for row in starts for col in starts)
    crop_start = crop_grid[(step // len(anchors)) % len(crop_grid)]
    row, col = crop_start
    labels = target[row : row + crop_size, col : col + crop_size].reshape(-1).long()
    probabilities = decode_anchor_patch(
        vae,
        latent,
        anchor,
        target_size=target_size,
        num_phases=num_phases,
        tile_overlap=tile_overlap,
        batch_size=decode_batch_size,
        crop_start=crop_start,
        crop_size=crop_size,
    )
    selected = probabilities.movedim(0, -1).reshape(-1, num_phases)
    pixel_loss = -selected.clamp_min(
        torch.finfo(selected.dtype).tiny
    ).log().gather(1, labels.unsqueeze(1))[:, 0]
    return torch.stack(
        [pixel_loss[labels == phase].mean() for phase in labels.unique()]
    ).mean()


def _validate(
    latent: torch.Tensor,
    vae: torch.nn.Module,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    num_phases: int,
    anchors: Sequence[AnchorSlice] | None,
    segment_anchors: bool,
    anchor_weight: float,
    fraction_targets: Mapping[int, float] | torch.Tensor | None,
    slice_fraction_weight: float,
    global_fraction_weight: float,
    tpc_targets: Mapping[int, torch.Tensor] | torch.Tensor | None,
    tpc_weight: float,
    sa_targets: Mapping[int, float] | torch.Tensor | None,
    sa_weight: float,
    diffusivity_targets: Mapping[int, float] | torch.Tensor | None,
    diffusivity_solver: ConductanceSolver | None,
    diffusivity_weight: float,
    sds_weight: float,
    continuity_weight: float,
    preservation_weight: float,
    residual_scale: float,
    checkpoint_every: int,
    decode_batch_size: int | None,
    temperature: float,
    progress: bool,
) -> None:
    if latent.ndim != 4:
        raise ValueError("latent must have shape [C, D, H, W].")
    require_finite("latent", latent)
    if not latent.is_floating_point():
        raise ValueError("latent must be floating point.")
    if latent.shape[0] != int(vae.latent_ch):
        raise ValueError("latent channel count must match vae.latent_ch.")
    if len(set(map(int, latent.shape[1:]))) != 1:
        raise ValueError("scale-up latent must be cubic.")
    if min(map(int, latent.shape[1:])) < int(vae.latent_size):
        raise ValueError("scale-up latent must contain a trained-size crop.")
    require_int("steps", steps)
    require_int("batch_size", batch_size)
    require_int("checkpoint_every", checkpoint_every)
    if steps < 0 or batch_size <= 0 or checkpoint_every <= 0:
        raise ValueError("scale optimization counts are invalid.")
    if decode_batch_size is not None:
        require_int("decode_batch_size", decode_batch_size)
        if decode_batch_size <= 0:
            raise ValueError("decode_batch_size must be positive or None.")
    if num_phases != getattr(vae, "num_phases", None):
        raise ValueError("num_phases must match vae.num_phases.")
    for name, value in (
        ("lr", lr),
        ("temperature", temperature),
        ("residual_scale", residual_scale),
    ):
        require_finite_number(name, value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")
    for name, value in (
        ("sds_weight", sds_weight),
        ("anchor_weight", anchor_weight),
        ("slice_fraction_weight", slice_fraction_weight),
        ("global_fraction_weight", global_fraction_weight),
        ("tpc_weight", tpc_weight),
        ("sa_weight", sa_weight),
        ("diffusivity_weight", diffusivity_weight),
        ("continuity_weight", continuity_weight),
        ("preservation_weight", preservation_weight),
    ):
        require_finite_number(name, value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if not isinstance(segment_anchors, bool):
        raise ValueError("segment_anchors must be a boolean.")
    if anchor_weight > 0.0 and not anchors:
        raise ValueError("scale anchors are required when anchor_weight is positive.")
    if (
        slice_fraction_weight > 0.0 or global_fraction_weight > 0.0
    ) and fraction_targets is None:
        raise ValueError("fraction_targets are required for fraction guidance.")
    if tpc_weight > 0.0 and tpc_targets is None:
        raise ValueError("tpc_targets are required when tpc_weight is positive.")
    if sa_weight > 0.0 and sa_targets is None:
        raise ValueError("sa_targets are required when sa_weight is positive.")
    if diffusivity_weight > 0.0 and (
        diffusivity_targets is None or diffusivity_solver is None
    ):
        raise ValueError("diffusivity targets and solver are required.")
    if not isinstance(progress, bool):
        raise ValueError("progress must be a boolean.")
