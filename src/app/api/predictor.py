from collections.abc import Sequence

import numpy as np
import torch

from src.modeling.phases.quantization import quantize_phase
from src.modeling.vae import get_downsample_factor
from src.pipelines.reconstruction.refinement import refine_axes
from src.modeling.diffusion import DiffusionSampler
from src.pipelines.guidance.conditioning.validation import validate_anchors
from src.pipelines.guidance.conditioning.latents import encode_anchors
from src.pipelines.guidance.conditioning.images import prepare_anchor_image
from src.pipelines.scaling.decoding import decode_large_volume
from src.pipelines.scaling.optimization import optimize_large_volume
from src.pipelines.scaling.refinement import refine_large_volume
from src.pipelines.scaling.sampling import sample_large_lmpdd
from src.pipelines.scaling.conditioning import (
    center_start,
    encode_scale_anchors,
    build_scale_targets,
    shift_anchor_slices,
)
from src.pipelines.guidance.optimization import optimize_slice, optimize_volume
from src.pipelines.guidance.joint_optimization import optimize_joint_volume
from src.pipelines.guidance.slicegan import generate_conditional_slicegan
from src.pipelines.guidance.conditioning.targets import build_sds_targets
from src.app.api.options import AnchorSlice, PredictOptions
from src.pipelines.reconstruction.volume import generate_initial_volume


from src.app.api.preparation import PredictionPrep

class Predictor(PredictionPrep):
    def __init__(
        self,
        vae: torch.nn.Module,
        diffusion_model: torch.nn.Module,
        ddpm,
        *,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.vae = vae.to(self.device)
        self.diffusion_model = diffusion_model.to(self.device)
        self.ddpm = ddpm
        self.sampler = DiffusionSampler(self.diffusion_model, self.ddpm, self.device)

    def predict(
        self,
        options: PredictOptions,
        *,
        anchors: Sequence[AnchorSlice] | None = None,
        target_images: Sequence[np.ndarray] | None = None,
        volume_size: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        volume_size = self._resolve_volume_size(
            anchors=anchors,
            volume_size=volume_size,
        )
        self._validate_anchors(anchors, volume_size)
        self._validate_inputs(options, target_images=target_images)

        if options.slicegan_steps > 0:
            volume, stats = self._run_slicegan(
                volume_size,
                options=options,
                anchors=anchors,
            )
            return quantize_phase(volume, options.num_phases), stats

        volume, stats = self._generate_base(
            volume_size,
            options=options,
            anchors=anchors,
        )

        if options.joint_3d_steps > 0:
            volume, joint_stats = self._run_joint_3d(
                volume,
                options=options,
                anchors=anchors,
                target_images=target_images,
            )
            stats = {**stats, **joint_stats}
        elif options.sds_steps > 0:
            volume, sds_stats = self._run_sds(
                volume,
                options=options,
                anchors=anchors,
                target_images=target_images,
            )

            stats = {**stats, **sds_stats}

        if options.refine_steps > 0:
            volume = self._refine(volume, options.refine_steps)

        if options.anchor_fit_steps > 0:
            volume, anchor_stats = self._fit_anchors(
                volume,
                options=options,
                anchors=anchors,
            )
            stats = {**stats, **anchor_stats}

        return quantize_phase(volume, options.num_phases), stats

    def _run_slicegan(
        self,
        volume_size: int,
        *,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if volume_size != self._get_image_size() or volume_size != 64:
            raise ValueError("conditional SliceGAN currently requires a 64³ volume.")
        if anchors is None or len(anchors) != 1:
            raise ValueError("conditional SliceGAN requires exactly one XY anchor.")
        anchor = anchors[0]
        if anchor.axis != 0:
            raise ValueError("conditional SliceGAN currently requires an XY anchor.")
        target = prepare_anchor_image(
            anchor.image,
            num_phases=options.num_phases,
            segment=options.anchor_segment,
        )[0, 0]
        return generate_conditional_slicegan(
            self.sampler,
            self.vae,
            anchor_image=target,
            anchor_index=anchor.index,
            num_phases=options.num_phases,
            steps=options.slicegan_steps,
            hybrid_steps=options.slicegan_hybrid_steps,
            condition_steps=options.slicegan_condition_steps,
            finetune_steps=options.slicegan_finetune_steps,
            seed=options.slicegan_seed,
            device=self.device,
        )

    def _run_joint_3d(
        self,
        volume: torch.Tensor,
        *,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
        target_images: Sequence[np.ndarray] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if volume.shape[0] != self._get_image_size():
            raise ValueError("joint 3D optimization currently requires the base volume size.")

        targets = self._build_targets(options, target_images)
        solver = targets.get("diffusivity_solver")
        if isinstance(solver, torch.nn.Module):
            solver = solver.to(self.device)

        return optimize_joint_volume(
            volume,
            self.vae,
            self.diffusion_model,
            self.ddpm,
            steps=options.joint_3d_steps,
            batch_size=options.joint_3d_batch_size,
            lr=options.joint_3d_lr,
            t_min=options.sds_t_min,
            t_max=self._resolve_t_max(options),
            num_phases=options.num_phases,
            anchors=anchors,
            anchor_segment=options.anchor_segment,
            sds_weight=options.sds_weight,
            anchor_weight=options.anchor_weight if anchors else 0.0,
            anchor_slab_radius=options.anchor_slab_radius,
            anchor_slab_weight=options.anchor_slab_weight,
            vf_targets=targets.get("vf_targets"),
            vf_weight=options.vf_weight,
            tpc_targets=targets.get("tpc_targets"),
            tpc_weight=options.tpc_weight,
            sa_targets=targets.get("sa_targets"),
            sa_weight=options.sa_weight,
            diffusivity_targets=targets.get("diffusivity_targets"),
            diffusivity_solver=solver,
            diffusivity_weight=options.diffusivity_weight,
            entropy_weight=options.joint_3d_entropy_weight,
            continuity_weight=options.joint_3d_continuity_weight,
            transition_weight=options.joint_3d_transition_weight,
            run_weight=options.joint_3d_run_weight,
            reference_images=target_images,
            patch_weight=options.joint_3d_patch_weight,
            texture_weight=options.joint_3d_texture_weight,
            interface_weight=options.joint_3d_interface_weight,
            discriminator_lr=options.joint_3d_discriminator_lr,
        )

    def _fit_anchors(
        self,
        volume: torch.Tensor,
        *,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not anchors:
            raise ValueError("anchors are required when anchor_fit_steps is positive.")

        if volume.shape[0] != self._get_image_size():
            raise ValueError("anchor fitting currently requires the base volume size.")

        updated = volume
        losses: list[torch.Tensor] = []

        for anchor in anchors:
            target = prepare_anchor_image(
                anchor.image,
                num_phases=options.num_phases,
                segment=options.anchor_segment,
            )[0, 0].to(device=self.device, dtype=volume.dtype)
            updated, fit_stats = optimize_slice(
                updated,
                self.vae,
                self.diffusion_model,
                self.ddpm,
                axis=anchor.axis,
                index=anchor.index,
                steps=options.anchor_fit_steps,
                lr=options.anchor_fit_lr,
                t_min=0,
                t_max=int(self.ddpm.num_timesteps),
                num_phases=options.num_phases,
                sds_weight=0.0,
                anchor_target=target,
                anchor_weight=options.anchor_weight,
            )
            losses.append(fit_stats["anchor"])

        return updated, {"anchor_fit": torch.stack(losses).mean()}

    def _generate_base(
        self,
        volume_size: int,
        *,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
    ) -> tuple[torch.Tensor, dict]:
        if volume_size == self._get_image_size():
            # Joint optimization applies the condition at the exact image-space
            # index. Injecting it into one coarse latent plane first shifts and
            # locally repeats the condition after trilinear decoding.
            base_anchors = None if options.joint_3d_steps > 0 else anchors
            anchor_latent, anchor_mask = encode_anchors(
                self.vae,
                base_anchors,
                num_phases=options.num_phases,
                segment=options.anchor_segment,
                device=self.device,
                spread_sigma=options.anchor_latent_sigma,
                peak_strength=options.anchor_latent_strength,
            )

            volume = generate_initial_volume(
                self.sampler,
                self.vae,
                size=self._get_image_size(),
                anchor_latent=anchor_latent,
                anchor_mask=anchor_mask,
                axis_consensus=options.lmpdd_axis_consensus,
            ).to(self.device)
            return volume, {}

        return self._generate_large(
            volume_size,
            options=options,
            anchors=anchors,
        )

    def _generate_large(
        self,
        volume_size: int,
        *,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        factor = get_downsample_factor(self.vae)
        tile_size = int(self.vae.latent_size)
        overlap = self._resolve_overlap(tile_size, None)
        latent_size = self._calc_latent_size(
            volume_size,
            factor=factor,
            tile_size=tile_size,
        )
        anchor_latent, anchor_mask = self._build_scale_latents(
            anchors,
            options=options,
            volume_size=volume_size,
            tile_overlap=overlap,
        )

        latent = sample_large_lmpdd(
            self.diffusion_model,
            self.ddpm,
            (
                int(self.vae.latent_ch),
                latent_size,
                latent_size,
                latent_size,
            ),
            tile_size=tile_size,
            tile_overlap=overlap,
            device=self.device,
            anchor_latent=anchor_latent,
            anchor_mask=anchor_mask,
        )
        volume = decode_large_volume(
            self.vae,
            latent,
            tile_overlap=overlap,
        )
        stats = {
            "volume_size": int(volume_size),
            "latent_size": latent_size,
            "tile_size": tile_size,
            "tile_overlap": overlap,
            "condition_start": center_start(
                volume_size=volume_size,
                base_size=self._get_image_size(),
            ),
        }
        return volume, stats

    def _refine(self, volume: torch.Tensor, steps: int) -> torch.Tensor:
        if volume.shape[0] == self._get_image_size():
            return refine_axes(
                volume,
                self.vae,
                steps=steps,
            )

        return refine_large_volume(
            volume,
            self.vae,
            steps=steps,
            tile_overlap=self._calc_refine_overlap(),
        )

    def _run_sds(
        self,
        volume: torch.Tensor,
        *,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
        target_images: Sequence[np.ndarray] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        kwargs = self._build_sds_args(
            options,
            anchors=anchors,
            target_images=target_images,
            volume_size=int(volume.shape[0]),
        )

        if volume.shape[0] != self._get_image_size():
            return optimize_large_volume(
                volume,
                self.vae,
                self.diffusion_model,
                self.ddpm,
                tile_overlap=self._calc_refine_overlap(),
                **kwargs,
            )

        kwargs.pop("anchor_targets", None)
        kwargs.pop("anchor_masks", None)
        kwargs.pop("descriptor_tile_size", None)
        kwargs["consensus_sweeps"] = (
            options.sds_consensus and options.sds_balanced_slices
        )
        kwargs["anchor_slab_radius"] = options.anchor_slab_radius
        kwargs["anchor_slab_weight"] = options.anchor_slab_weight
        return optimize_volume(
            volume,
            self.vae,
            self.diffusion_model,
            self.ddpm,
            **kwargs,
        )

    def _build_sds_args(
        self,
        options: PredictOptions,
        *,
        anchors: Sequence[AnchorSlice] | None,
        target_images: Sequence[np.ndarray] | None,
        volume_size: int,
    ) -> dict[str, object]:
        targets = self._build_targets(options, target_images)
        solver = targets.get("diffusivity_solver")

        if isinstance(solver, torch.nn.Module):
            solver = solver.to(self.device)

        sds_anchors = anchors
        anchor_targets = None
        anchor_masks = None
        slice_schedule = None

        if self._has_scale_anchor(anchors, volume_size):
            anchor_targets, anchor_masks = build_scale_targets(
                self.vae,
                anchors,
                volume_size=volume_size,
                base_size=self._get_image_size(),
                num_phases=options.num_phases,
                segment=options.anchor_segment,
                device=self.device,
                dtype=torch.float32,
                downsample_factor=get_downsample_factor(self.vae),
            )

            sds_anchors = None
            slice_schedule = self._build_anchor_schedule(
                anchors,
                steps=options.sds_steps,
                batch_size=options.sds_batch_size,
                volume_size=volume_size,
            )
        elif options.sds_balanced_slices:
            slice_schedule = self._build_balanced_schedule(
                steps=options.sds_steps,
                batch_size=options.sds_batch_size,
                volume_size=volume_size,
            )

        return {
            "steps": options.sds_steps,
            "slice_steps": options.sds_slice_steps,
            "sds_batch_size": options.sds_batch_size,
            "lr": options.sds_lr,
            "t_min": options.sds_t_min,
            "t_max": self._resolve_t_max(options),
            "num_phases": options.num_phases,
            "slice_schedule": slice_schedule,
            "anchors": sds_anchors,
            "anchor_targets": anchor_targets,
            "anchor_masks": anchor_masks,
            "anchor_segment": options.anchor_segment,
            "sds_weight": options.sds_weight,
            "anchor_weight": options.anchor_weight if anchors else 0.0,
            "vf_targets": targets.get("vf_targets"),
            "vf_weight": options.vf_weight,
            "tpc_targets": targets.get("tpc_targets"),
            "tpc_weight": options.tpc_weight,
            "sa_targets": targets.get("sa_targets"),
            "sa_weight": options.sa_weight,
            "diffusivity_targets": targets.get("diffusivity_targets"),
            "diffusivity_solver": solver,
            "diffusivity_weight": options.diffusivity_weight,
            "descriptor_tile_size": self._resolve_tile_size(
                options,
                target_images=target_images,
                volume_size=volume_size,
            ),
        }

    def _build_targets(
        self,
        options: PredictOptions,
        target_images: Sequence[np.ndarray] | None,
    ) -> dict[str, torch.Tensor | torch.nn.Module]:
        if not self._uses_targets(options):
            return {}

        return build_sds_targets(
            list(target_images or []),
            num_phases=options.num_phases,
            segment=options.target_segment,
            use_vf=options.vf_weight > 0.0,
            use_tpc=options.tpc_weight > 0.0,
            use_sa=options.sa_weight > 0.0,
            use_diffusivity=options.diffusivity_weight > 0.0,
            diffusivity_size=options.diffusivity_size,
            diffusivity_low_cond=options.diffusivity_low_cond,
        )

    def _validate_inputs(
        self,
        options: PredictOptions,
        *,
        target_images: Sequence[np.ndarray] | None,
    ) -> None:
        uses_targets = self._uses_targets(options)

        if options.slicegan_steps > 0 and uses_targets:
            raise ValueError(
                "descriptor target losses are not used by conditional SliceGAN."
            )

        if uses_targets and options.sds_steps <= 0 and options.joint_3d_steps <= 0:
            raise ValueError(
                "target losses require sds_steps or joint_3d_steps to be positive."
            )

        if uses_targets and not target_images:
            raise ValueError("target_images are required when target losses are enabled.")

        if options.sds_steps > 0 or options.joint_3d_steps > 0:
            self._resolve_t_max(options)

    def _get_image_size(self) -> int:
        return int(self.vae.image_size)
