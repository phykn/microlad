from collections.abc import Sequence

import numpy as np
import torch

from src.app.api.options import PredictOptions
from src.app.api.prepare import (
    anchor_size,
    prepare_prediction,
    uses_image_targets,
)
from src.modeling.diffusion import DDPMProcess, DiffusionSampler, TimeUNet
from src.modeling.phases.quantization import quantize_phase
from src.modeling.vae import PatchVAE, get_downsample_factor
from src.pipelines.guidance.conditioning.images import (
    prepare_anchor_image,
    prepare_volume_anchors,
)
from src.pipelines.guidance.conditioning.latents import encode_anchors
from src.pipelines.guidance.conditioning.model import AnchorSlice, VolumeAnchor
from src.pipelines.guidance.conditioning.targets import (
    SDSTargets,
    build_sds_targets,
    prepare_target_images,
)
from src.pipelines.guidance.metrics.diagnostics import evaluate_phase_volume
from src.pipelines.guidance.joint.optimize import optimize_joint_volume
from src.pipelines.guidance.sds.optimize import optimize_volume
from src.pipelines.guidance.sds.slice import optimize_slice
from src.pipelines.guidance.sds.schedule import (
    build_anchor_schedule,
    build_balanced_schedule,
)
from src.pipelines.guidance.slicegan.generate import generate_conditional_slicegan
from src.pipelines.reconstruction.refinement import refine_volume
from src.pipelines.reconstruction.slices import extract_slice
from src.pipelines.reconstruction.volume import generate_initial_volume
from src.pipelines.scaling.conditioning import (
    build_scale_targets,
    center_start,
    encode_scale_anchors,
)
from src.pipelines.scaling.decoding import decode_large_volume
from src.pipelines.scaling.optimization import optimize_large_volume
from src.pipelines.scaling.refinement import refine_large_volume
from src.pipelines.scaling.sampling import sample_large_lmpdd


class Predictor:
    """Generates conditional categorical 3D volumes.

    Args:
        vae: Trained categorical VAE.
        diffusion_model: Trained latent diffusion denoiser.
        ddpm: Diffusion process paired with the denoiser.
        device: Device used for prediction.
    """

    def __init__(
        self,
        vae: PatchVAE,
        diffusion_model: TimeUNet,
        ddpm: DDPMProcess,
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
        """Generates a categorical 3D volume.

        Args:
            options: Generation and conditioning settings.
            anchors: Conditional slices at specified axes and indices.
            target_images: Reference images used by descriptor losses.
            volume_size: Cubic output size, or None to infer it.

        Returns:
            Generated uint8 volume and prediction statistics.
        """

        image_size = int(self.vae.image_size)
        target_labels = (
            None
            if target_images is None
            else prepare_target_images(
                target_images,
                num_phases=options.num_phases,
                segment=options.targets.segment,
            )
        )
        volume_size, descriptor_tile_size, t_max = prepare_prediction(
            options,
            anchors,
            target_labels,
            volume_size,
            image_size,
            int(self.ddpm.num_timesteps),
        )
        volume_anchors = prepare_volume_anchors(
            anchors,
            volume_size=volume_size,
            num_phases=options.num_phases,
            segment=options.segment_anchors,
            device=self.device,
            intersection_tolerance=(
                options.slicegan.intersection_tolerance
                if options.slicegan is not None
                else 0.0
            ),
        )

        if options.slicegan is not None:
            volume, stats = self._run_slicegan(
                volume_size,
                options=options,
                anchors=volume_anchors,
            )
            volume = quantize_phase(volume, options.num_phases)
            final_stats = evaluate_phase_volume(
                volume,
                num_phases=options.num_phases,
                target_fraction=(
                    None
                    if options.phase_fractions is None
                    else torch.tensor(
                        options.phase_fractions,
                        device=volume.device,
                        dtype=torch.float32,
                    )
                ),
                anchors=volume_anchors,
            )
            stats.update(
                {f"final_{key}": value for key, value in final_stats.items()}
            )
            return volume, stats

        volume, base_stats = self._generate_base(
            volume_size,
            options=options,
            anchors=anchors,
        )
        stats: dict[str, torch.Tensor | int] = dict(base_stats)

        if options.joint.steps > 0:
            assert t_max is not None
            volume, joint_stats = self._run_joint(
                volume,
                options=options,
                anchors=anchors,
                target_labels=target_labels,
                t_max=t_max,
            )
            stats.update(joint_stats)
        elif options.sds.steps > 0:
            assert t_max is not None
            volume, sds_stats = self._run_sds(
                volume,
                options=options,
                anchors=anchors,
                target_labels=target_labels,
                descriptor_tile_size=descriptor_tile_size,
                t_max=t_max,
            )

            stats.update(sds_stats)

        if options.refine.steps > 0:
            if volume.shape[0] == image_size:
                volume = refine_volume(
                    volume,
                    self.vae,
                    steps=options.refine.steps,
                    batch_size=options.refine.batch_size,
                )
            else:
                volume = refine_large_volume(
                    volume,
                    self.vae,
                    steps=options.refine.steps,
                    tile_overlap=_tile_overlap(
                        image_size,
                        options.scale.overlap,
                    ),
                    tile_batch_size=options.scale.batch_size,
                )

        if options.diffusion_anchor.fit_steps > 0:
            volume, anchor_stats = self._fit_anchors(
                volume,
                options=options,
                anchors=anchors,
            )
            stats.update(anchor_stats)

        volume = quantize_phase(volume, options.num_phases)
        references = (
            None
            if target_labels is None
            else torch.nn.functional.one_hot(
                target_labels.to(dtype=torch.long),
                num_classes=options.num_phases,
            )
            .movedim(-1, 1)
            .float()
        )
        target_fraction = (
            torch.tensor(
                options.phase_fractions,
                device=volume.device,
                dtype=torch.float32,
            )
            if options.phase_fractions is not None
            else (
                None
                if references is None
                else references.mean(dim=(0, 2, 3))
            )
        )
        final_stats = evaluate_phase_volume(
            volume,
            num_phases=options.num_phases,
            references=references,
            target_fraction=target_fraction,
            anchors=volume_anchors,
        )
        stats.update({f"final_{key}": value for key, value in final_stats.items()})
        return volume, stats

    def _run_slicegan(
        self,
        volume_size: int,
        options: PredictOptions,
        anchors: Sequence[VolumeAnchor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not anchors:
            raise ValueError("conditional SliceGAN requires at least one anchor.")
        config = options.slicegan
        if config is None:
            raise ValueError("slicegan options are required.")
        target_fraction = (
            None
            if options.phase_fractions is None
            else torch.tensor(
                options.phase_fractions,
                device=self.device,
                dtype=torch.float32,
            )
        )
        return generate_conditional_slicegan(
            self.sampler,
            self.vae,
            anchors=anchors,
            target_fraction=target_fraction,
            phase_fraction_tolerance=options.phase_fraction_tolerance,
            volume_size=volume_size,
            num_phases=options.num_phases,
            config=config,
            device=self.device,
            scale_batch_size=options.scale.batch_size,
        )

    def _run_joint(
        self,
        volume: torch.Tensor,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
        target_labels: torch.Tensor | None,
        t_max: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if volume.shape[0] != int(self.vae.image_size):
            raise ValueError(
                "joint 3D optimization currently requires the base volume size."
            )

        targets = self._build_targets(options, target_labels)
        solver = targets.get("diffusivity_solver")

        return optimize_joint_volume(
            volume,
            self.vae,
            self.diffusion_model,
            self.ddpm,
            steps=options.joint.steps,
            batch_size=options.joint.batch_size,
            lr=options.joint.learning_rate,
            t_min=options.prior.t_min,
            t_max=t_max,
            num_phases=options.num_phases,
            anchors=anchors,
            segment_anchors=options.segment_anchors,
            sds_weight=options.prior.weight,
            anchor_weight=options.diffusion_anchor.weight if anchors else 0.0,
            anchor_slab_radius=options.diffusion_anchor.slab_radius,
            anchor_slab_weight=options.diffusion_anchor.slab_weight,
            vf_targets=targets.get("vf_targets"),
            vf_weight=(
                options.targets.vf_weight
                if options.targets.vf_weight > 0.0 or options.phase_fractions is None
                else 1.0
            ),
            tpc_targets=targets.get("tpc_targets"),
            tpc_weight=options.targets.tpc_weight,
            sa_targets=targets.get("sa_targets"),
            sa_weight=options.targets.surface_area_weight,
            diffusivity_targets=targets.get("diffusivity_targets"),
            diffusivity_solver=solver,
            diffusivity_weight=options.targets.diffusivity_weight,
            entropy_weight=options.joint.entropy_weight,
            continuity_weight=options.joint.continuity_weight,
            transition_weight=options.joint.transition_weight,
            run_weight=options.joint.run_weight,
            reference_labels=target_labels,
            patch_weight=options.joint.patch_weight,
            texture_weight=options.joint.texture_weight,
            interface_weight=options.joint.interface_weight,
            discriminator_lr=options.joint.discriminator_lr,
        )

    def _fit_anchors(
        self,
        volume: torch.Tensor,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not anchors:
            raise ValueError(
                "anchors are required when diffusion_anchor.fit_steps is positive."
            )

        if volume.shape[0] != int(self.vae.image_size):
            raise ValueError("anchor fitting currently requires the base volume size.")

        updated = volume
        losses: list[torch.Tensor] = []

        for anchor in anchors:
            target = prepare_anchor_image(
                anchor.image,
                num_phases=options.num_phases,
                segment=options.segment_anchors,
            )[0, 0].to(device=self.device, dtype=volume.dtype)
            updated, fit_stats = optimize_slice(
                updated,
                self.vae,
                self.diffusion_model,
                self.ddpm,
                axis=anchor.axis,
                index=anchor.index,
                steps=options.diffusion_anchor.fit_steps,
                lr=options.diffusion_anchor.fit_lr,
                t_min=0,
                t_max=int(self.ddpm.num_timesteps),
                num_phases=options.num_phases,
                sds_weight=0.0,
                anchor_target=target,
                anchor_weight=options.diffusion_anchor.weight,
            )
            losses.append(fit_stats["anchor"])

        mismatches = []
        for anchor in anchors:
            target = prepare_anchor_image(
                anchor.image,
                num_phases=options.num_phases,
                segment=options.segment_anchors,
            )[0, 0].to(device=updated.device)
            actual = extract_slice(updated, int(anchor.axis), int(anchor.index))
            mismatches.append((actual.round() != target.round()).float().mean())
        mismatch_tensor = torch.stack(mismatches)
        return updated, {
            "anchor_fit_history": torch.stack(losses).mean(),
            "anchor_mismatches": mismatch_tensor,
            "anchor_mismatch": mismatch_tensor.mean(),
            "anchor_max_mismatch": mismatch_tensor.max(),
        }

    def _generate_base(
        self,
        volume_size: int,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        if volume_size == int(self.vae.image_size):
            # Joint optimization applies the condition at the exact image-space
            # index. Injecting it into one coarse latent plane first shifts and
            # locally repeats the condition after trilinear decoding.
            base_anchors = None if options.joint.steps > 0 else anchors
            anchor_latent, anchor_mask = encode_anchors(
                self.vae,
                base_anchors,
                num_phases=options.num_phases,
                segment=options.segment_anchors,
                device=self.device,
                spread_sigma=options.diffusion_anchor.latent_sigma,
                peak_strength=options.diffusion_anchor.latent_strength,
            )

            volume = generate_initial_volume(
                self.sampler,
                self.vae,
                anchor_latent=anchor_latent,
                anchor_mask=anchor_mask,
                axis_consensus=options.diffusion_anchor.axis_consensus,
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
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        factor = get_downsample_factor(self.vae)
        tile_size = int(self.vae.latent_size)
        overlap = _tile_overlap(tile_size, options.scale.overlap)
        if volume_size % factor != 0:
            raise ValueError("volume_size must be divisible by VAE downsample factor.")
        latent_size = volume_size // factor
        if latent_size < tile_size:
            raise ValueError("volume_size must be at least vae.image_size.")

        condition_size = anchor_size(anchors)
        if anchors and condition_size in (int(self.vae.image_size), volume_size):
            anchor_latent, anchor_mask = encode_scale_anchors(
                self.vae,
                anchors,
                volume_size=volume_size,
                num_phases=options.num_phases,
                segment=options.segment_anchors,
                device=self.device,
                tile_overlap=overlap,
            )
        else:
            anchor_latent, anchor_mask = None, None

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
            batch_size=options.scale.batch_size,
            anchor_latent=anchor_latent,
            anchor_mask=anchor_mask,
        )
        volume = decode_large_volume(
            self.vae,
            latent,
            tile_overlap=overlap,
            batch_size=options.scale.batch_size,
        )
        stats = {
            "volume_size": int(volume_size),
            "latent_size": latent_size,
            "tile_size": tile_size,
            "tile_overlap": overlap,
            "condition_start": center_start(
                volume_size=volume_size,
                base_size=int(self.vae.image_size),
            ),
        }
        return volume, stats

    def _run_sds(
        self,
        volume: torch.Tensor,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
        target_labels: torch.Tensor | None,
        descriptor_tile_size: int | None,
        t_max: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        volume_size = int(volume.shape[0])
        image_size = int(self.vae.image_size)
        targets = self._build_targets(options, target_labels)
        solver = targets.get("diffusivity_solver")

        sds_anchors = anchors
        anchor_targets = None
        anchor_masks = None
        slice_schedule = None
        if anchors and volume_size > image_size and anchor_size(anchors) == image_size:
            anchor_targets, anchor_masks = build_scale_targets(
                self.vae,
                anchors,
                volume_size=volume_size,
                base_size=image_size,
                num_phases=options.num_phases,
                segment=options.segment_anchors,
                device=self.device,
                dtype=torch.float32,
                downsample_factor=get_downsample_factor(self.vae),
            )
            sds_anchors = None
            slice_schedule = build_anchor_schedule(
                anchors,
                steps=options.sds.steps,
                batch_size=options.sds.batch_size,
                volume_size=volume_size,
                base_size=image_size,
                downsample_factor=get_downsample_factor(self.vae),
                device=self.device,
            )
        elif options.sds.balanced_slices:
            slice_schedule = build_balanced_schedule(
                steps=options.sds.steps,
                batch_size=options.sds.batch_size,
                volume_size=volume_size,
            )

        kwargs = {
            "steps": options.sds.steps,
            "slice_steps": options.sds.slice_steps,
            "batch_size": options.sds.batch_size,
            "lr": options.sds.learning_rate,
            "t_min": options.prior.t_min,
            "t_max": t_max,
            "num_phases": options.num_phases,
            "slice_schedule": slice_schedule,
            "anchors": sds_anchors,
            "anchor_targets": anchor_targets,
            "anchor_masks": anchor_masks,
            "segment_anchors": options.segment_anchors,
            "sds_weight": options.prior.weight,
            "anchor_weight": options.diffusion_anchor.weight if anchors else 0.0,
            "vf_targets": targets.get("vf_targets"),
            "vf_weight": (
                options.targets.vf_weight
                if options.targets.vf_weight > 0.0 or options.phase_fractions is None
                else 1.0
            ),
            "tpc_targets": targets.get("tpc_targets"),
            "tpc_weight": options.targets.tpc_weight,
            "sa_targets": targets.get("sa_targets"),
            "sa_weight": options.targets.surface_area_weight,
            "diffusivity_targets": targets.get("diffusivity_targets"),
            "diffusivity_solver": solver,
            "diffusivity_weight": options.targets.diffusivity_weight,
            "descriptor_tile_size": descriptor_tile_size,
        }

        if volume_size != image_size:
            return optimize_large_volume(
                volume,
                self.vae,
                self.diffusion_model,
                self.ddpm,
                tile_overlap=_tile_overlap(image_size, options.scale.overlap),
                **kwargs,
            )

        kwargs.pop("anchor_targets", None)
        kwargs.pop("anchor_masks", None)
        kwargs.pop("descriptor_tile_size", None)
        kwargs["consensus_sweeps"] = (
            options.sds.consensus_sweeps and options.sds.balanced_slices
        )
        kwargs["anchor_slab_radius"] = options.diffusion_anchor.slab_radius
        kwargs["anchor_slab_weight"] = options.diffusion_anchor.slab_weight
        return optimize_volume(
            volume,
            self.vae,
            self.diffusion_model,
            self.ddpm,
            **kwargs,
        )

    def _build_targets(
        self,
        options: PredictOptions,
        target_labels: torch.Tensor | None,
    ) -> SDSTargets:
        targets: SDSTargets = {}
        if uses_image_targets(options):
            targets.update(
                build_sds_targets(
                    target_labels,
                    num_phases=options.num_phases,
                    use_vf=(
                        options.targets.vf_weight > 0.0
                        and options.phase_fractions is None
                    ),
                    use_tpc=options.targets.tpc_weight > 0.0,
                    use_sa=options.targets.surface_area_weight > 0.0,
                    use_diffusivity=options.targets.diffusivity_weight > 0.0,
                    diffusivity_grid_size=options.targets.diffusivity_grid_size,
                    low_phase_conductivity=options.targets.low_phase_conductivity,
                )
            )
        if options.phase_fractions is not None:
            targets["vf_targets"] = torch.tensor(
                options.phase_fractions,
                device=self.device,
                dtype=torch.float32,
            )
        solver = targets.get("diffusivity_solver")
        if solver is not None:
            targets["diffusivity_solver"] = solver.to(self.device)
        return targets


def _tile_overlap(tile_size: int, ratio: float) -> int:
    if tile_size <= 1 or ratio == 0.0:
        return 0
    return max(int(tile_size * ratio), 1)
