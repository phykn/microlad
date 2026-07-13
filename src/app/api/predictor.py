from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.app.api.options import PredictOptions
from src.app.api.prepare import (
    anchor_size,
    prepare_prediction,
    uses_descriptor_targets,
)
from src.modeling.diffusion import DDPMProcess, DiffusionSampler, TimeUNet
from src.modeling.critic import LatentCritic
from src.modeling.phases.quantization import quantize_phase
from src.modeling.vae import PatchVAE, get_downsample_factor
from src.pipelines.guidance.critic.data import encode_refs, merge_refs
from src.pipelines.guidance.critic.train import train_critic
from src.pipelines.guidance.conditioning.images import prepare_volume_anchors
from src.pipelines.guidance.conditioning.model import AnchorSlice, VolumeAnchor
from src.pipelines.guidance.conditioning.targets import (
    SDSTargets,
    build_sds_targets,
    prepare_target_images,
)
from src.pipelines.guidance.metrics.diagnostics import evaluate_phase_volume
from src.pipelines.guidance.finalize.quality import (
    check_quality,
    reference_transition,
)
from src.pipelines.guidance.finalize.select import (
    select_volume,
)
from src.pipelines.guidance.joint.optimize import optimize_latent
from src.pipelines.scaling.schedule import (
    build_anchor_schedule,
    build_balanced_schedule,
)
from src.pipelines.reconstruction.volume import sample_latent
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
            intersection_tolerance=0.0,
        )

        if volume_size == image_size:
            assert t_max is not None or options.joint.steps == 0
            return self._predict_base(
                options,
                anchors=anchors,
                volume_anchors=volume_anchors,
                target_labels=target_labels,
                t_max=t_max,
            )

        volume, base_stats = self._generate_large(
            volume_size,
            options=options,
            anchors=anchors,
        )
        stats: dict[str, torch.Tensor | int] = dict(base_stats)

        if options.scale.steps > 0:
            assert t_max is not None
            volume, joint_stats = self._refine_large(
                volume,
                options=options,
                anchors=anchors,
                target_labels=target_labels,
                descriptor_tile_size=descriptor_tile_size,
                t_max=t_max,
            )
            stats.update(joint_stats)
        refine_steps = max(options.refine.candidates)
        if refine_steps > 0:
            volume = refine_large_volume(
                volume,
                self.vae,
                steps=refine_steps,
                tile_overlap=_tile_overlap(
                    image_size,
                    options.scale.overlap,
                ),
                tile_batch_size=options.scale.batch_size,
            )
            stats["selected_refine_steps"] = refine_steps

        volume = quantize_phase(volume, options.num_phases)
        reference_labels = self._reference_labels(volume_anchors, target_labels)
        references = (
            None
            if reference_labels is None
            else F.one_hot(
                reference_labels.long(),
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
            else None
        )
        final_stats = evaluate_phase_volume(
            volume,
            num_phases=options.num_phases,
            references=references,
            target_fraction=target_fraction,
            anchors=volume_anchors,
        )
        stats.update({f"final_{key}": value for key, value in final_stats.items()})
        passed, errors = check_quality(
            final_stats,
            calibration_valid=True,
            changed=volume.new_zeros((), dtype=torch.float32),
            target_transition=reference_transition(references),
            phase_fraction_tolerance=options.phase_fraction_tolerance,
            quality=options.quality,
        )
        stats.update({f"quality_{key}": value for key, value in errors.items()})
        stats["quality_passed"] = torch.tensor(passed, device=volume.device)
        if options.quality.strict and not passed:
            failed = [
                name
                for name, value in errors.items()
                if float(value.item()) > 0.0
            ]
            raise RuntimeError(
                "scale-up prediction failed the final quality gate: "
                + ", ".join(failed)
            )
        return volume, stats

    def _predict_base(
        self,
        options: PredictOptions,
        *,
        anchors: Sequence[AnchorSlice] | None,
        volume_anchors: Sequence[VolumeAnchor],
        target_labels: torch.Tensor | None,
        t_max: int | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
        latent = sample_latent(
            self.sampler,
            self.vae,
            progress=options.progress,
        ).to(self.device)
        references = self._reference_labels(volume_anchors, target_labels)
        critic = None
        stats: dict[str, torch.Tensor | int] = {}
        if options.critic.steps > 0:
            if references is None:
                raise ValueError("critic training requires target images or anchors.")
            critic, critic_stats = self._train_critic(
                latent,
                references,
                options=options,
            )
            stats.update(critic_stats)

        candidates, joint_stats = self._run_joint(
            latent,
            options=options,
            anchors=anchors,
            target_labels=target_labels,
            critic=critic,
            t_max=(int(self.ddpm.num_timesteps) if t_max is None else t_max),
        )
        stats.update(joint_stats)
        target_fraction = (
            None
            if options.phase_fractions is None
            else torch.tensor(
                options.phase_fractions,
                device=self.device,
                dtype=torch.float32,
            )
        )
        reference_probabilities = (
            None
            if references is None
            else F.one_hot(
                references.long(),
                num_classes=options.num_phases,
            ).movedim(-1, 1).float()
        )
        volume, final_stats = select_volume(
            self.vae,
            candidates,
            candidate_steps=joint_stats["joint_candidate_steps"].tolist(),
            num_phases=options.num_phases,
            target_fraction=target_fraction,
            phase_fraction_tolerance=options.phase_fraction_tolerance,
            anchors=volume_anchors,
            references=reference_probabilities,
            refine=options.refine,
            quality=options.quality,
        )
        stats.update(final_stats)
        return quantize_phase(volume, options.num_phases), stats

    def _train_critic(
        self,
        latent: torch.Tensor,
        references: torch.Tensor,
        *,
        options: PredictOptions,
    ) -> tuple[LatentCritic, dict[str, torch.Tensor]]:
        real_bank = encode_refs(
            self.vae,
            references,
            batch_size=options.critic.batch_size,
        )
        fake = [latent]
        for _ in range(options.critic.candidate_count - 1):
            fake.append(
                sample_latent(
                    self.sampler,
                    self.vae,
                    progress=options.progress,
                ).to(self.device)
            )
        fake_volumes = torch.stack(fake)
        critic = LatentCritic(int(self.vae.latent_ch)).to(self.device)
        stats = train_critic(
            critic,
            real_bank,
            fake_volumes,
            config=options.critic,
            progress=options.progress,
        )
        margin = float(stats["critic_validation_margin"].item())
        damage_margin = float(stats["critic_damage_margin"].item())
        finite_gradient = bool(stats["critic_input_gradient_finite"].item())
        if (
            not np.isfinite(margin)
            or not np.isfinite(damage_margin)
            or margin <= options.critic.min_margin
            or damage_margin <= options.critic.min_damage_margin
            or not finite_gradient
        ):
            raise RuntimeError(
                "latent critic failed held-out margin or gradient validation."
            )
        return critic, stats

    def _reference_labels(
        self,
        anchors: Sequence[VolumeAnchor],
        target_labels: torch.Tensor | None,
    ) -> torch.Tensor | None:
        anchor_labels = (
            torch.stack([anchor.image for anchor in anchors]) if anchors else None
        )
        return merge_refs(
            None if anchor_labels is None else anchor_labels.to(self.device),
            None if target_labels is None else target_labels.to(self.device),
        )

    def _run_joint(
        self,
        latent: torch.Tensor,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
        target_labels: torch.Tensor | None,
        critic: LatentCritic | None,
        t_max: int,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        targets = self._build_targets(options, target_labels)
        solver = targets.get("diffusivity_solver")

        return optimize_latent(
            latent,
            self.vae,
            self.diffusion_model,
            self.ddpm,
            steps=options.joint.steps,
            batch_size=options.joint.batch_size,
            decode_batch_size=options.joint.decode_batch_size,
            lr=options.joint.learning_rate,
            t_min=options.prior.t_min,
            t_max=t_max,
            num_phases=options.num_phases,
            anchors=anchors,
            segment_anchors=options.segment_anchors,
            sds_weight=options.prior.weight,
            critic=critic,
            critic_weight=options.critic.weight,
            anchor_weight=options.joint.anchor_weight if anchors else 0.0,
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
            continuity_weight=options.joint.continuity_weight,
            residual_scale=options.joint.residual_scale,
            preservation_weight=options.joint.preservation_weight,
            checkpoint_every=options.joint.checkpoint_every,
            progress=options.progress,
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
            progress=options.progress,
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

    def _refine_large(
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
        steps = options.scale.steps
        batch_size = options.scale.batch_size
        learning_rate = options.scale.learning_rate
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
                steps=steps,
                batch_size=batch_size,
                volume_size=volume_size,
                base_size=image_size,
                downsample_factor=get_downsample_factor(self.vae),
                device=self.device,
            )
        elif options.scale.balanced_slices:
            slice_schedule = build_balanced_schedule(
                steps=steps,
                batch_size=batch_size,
                volume_size=volume_size,
            )

        kwargs = {
            "steps": steps,
            "slice_steps": options.scale.slice_steps,
            "batch_size": batch_size,
            "lr": learning_rate,
            "t_min": options.prior.t_min,
            "t_max": t_max,
            "num_phases": options.num_phases,
            "slice_schedule": slice_schedule,
            "anchors": sds_anchors,
            "anchor_targets": anchor_targets,
            "anchor_masks": anchor_masks,
            "segment_anchors": options.segment_anchors,
            "sds_weight": options.prior.weight,
            "anchor_weight": options.scale.anchor_weight if anchors else 0.0,
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
            "progress": options.progress,
        }

        return optimize_large_volume(
            volume,
            self.vae,
            self.diffusion_model,
            self.ddpm,
            tile_overlap=_tile_overlap(image_size, options.scale.overlap),
            **kwargs,
        )

    def _build_targets(
        self,
        options: PredictOptions,
        target_labels: torch.Tensor | None,
    ) -> SDSTargets:
        targets: SDSTargets = {}
        if uses_descriptor_targets(options):
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
