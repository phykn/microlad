import warnings
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
from src.pipelines.guidance.conditioning.images import prepare_volume_anchors
from src.pipelines.guidance.conditioning.latents import encode_anchors
from src.pipelines.guidance.conditioning.model import AnchorSlice, VolumeAnchor
from src.pipelines.guidance.conditioning.prepare import build_volume_anchor_mask
from src.pipelines.guidance.conditioning.targets import (
    DescriptorTargets,
    build_descriptor_targets,
    prepare_target_images,
)
from src.pipelines.finalize.select import (
    select_latent_volume,
    select_probability_volume,
)
from src.pipelines.guidance.joint.optimize import optimize_latent
from src.pipelines.reconstruction.volume import sample_latent
from src.pipelines.scaling.conditioning import (
    center_start,
    encode_scale_anchors,
)
from src.pipelines.scaling.decoding import decode_large_volume_probabilities
from src.pipelines.scaling.optimize import optimize_large_latent
from src.pipelines.scaling.refine import refine_large_probabilities
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
        critic: LatentCritic | None = None,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.vae = vae.to(self.device)
        self.diffusion_model = diffusion_model.to(self.device)
        self.critic = None if critic is None else critic.to(self.device).eval()
        if self.critic is not None:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
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
        if options.critic.weight > 0.0 and self.critic is None:
            raise ValueError(
                "critic guidance requires models.gan_run_dir in predict.yaml."
            )
        target_labels = (
            None
            if target_images is None
            else prepare_target_images(
                target_images,
                num_phases=options.num_phases,
                segment=options.targets.segment,
            )
        )
        volume_size, _, t_max = prepare_prediction(
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
        anchor_labels = (
            torch.stack([anchor.image for anchor in volume_anchors])
            if volume_anchors
            else None
        )
        guidance_labels = target_labels
        if guidance_labels is None and uses_descriptor_targets(options):
            guidance_labels = anchor_labels

        if volume_size == image_size:
            assert t_max is not None or options.joint.steps == 0
            return self._predict_base(
                options,
                anchors=anchors,
                volume_anchors=volume_anchors,
                target_labels=target_labels,
                t_max=t_max,
            )

        if options.joint.steps > 0:
            warnings.warn(
                "joint settings are base-size only; scale settings will guide "
                "this large prediction.",
                RuntimeWarning,
                stacklevel=2,
            )

        latent, base_stats = self._generate_large(
            volume_size,
            options=options,
            anchors=anchors,
        )
        stats: dict[str, torch.Tensor | int] = dict(base_stats)
        stats["critic_enabled"] = torch.tensor(
            self.critic is not None and options.critic.weight > 0.0,
            device=self.device,
        )

        if options.scale.steps > 0:
            assert t_max is not None
            latent_candidates, joint_stats = self._refine_large(
                latent,
                options=options,
                anchors=anchors,
                target_labels=guidance_labels,
                t_max=t_max,
            )
            stats.update(joint_stats)
            scale_steps = joint_stats["scale_candidate_steps"].tolist()
        else:
            latent_candidates = (latent,)
            scale_steps = [0]
            stats["scale_steps"] = torch.tensor(0, device=self.device)
            stats["scale_candidate_steps"] = torch.tensor([0], device=self.device)

        reference_labels = target_labels if target_labels is not None else anchor_labels
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
        target_fraction = self._resolve_fraction(options, guidance_labels)
        anchor_mask = build_volume_anchor_mask(
            (volume_size, volume_size, volume_size),
            volume_anchors,
            device=self.device,
        )
        base_std = latent_candidates[0].std().clamp_min(1e-6)
        probability_candidates = []
        candidate_steps = []
        refine_steps = []
        candidate_deltas = []
        for scale_step, candidate in zip(
            scale_steps,
            latent_candidates,
            strict=True,
        ):
            probabilities = decode_large_volume_probabilities(
                self.vae,
                candidate,
                tile_overlap=_tile_overlap(
                    int(self.vae.latent_size),
                    options.scale.overlap,
                ),
                batch_size=options.scale.decode_batch_size,
            )
            refined = refine_large_probabilities(
                probabilities,
                self.vae,
                candidates=options.refine.candidates,
                tile_overlap=_tile_overlap(image_size, options.scale.overlap),
                tile_batch_size=options.scale.decode_batch_size,
                strength=options.refine.strength,
                anchor_strength=options.refine.anchor_strength,
                anchor_mask=anchor_mask,
            )
            delta = (candidate - latent_candidates[0]).square().mean().sqrt() / base_std
            for steps, values in zip(
                options.refine.candidates,
                refined,
                strict=True,
            ):
                probability_candidates.append(values)
                candidate_steps.append(int(scale_step))
                refine_steps.append(int(steps))
                candidate_deltas.append(delta)

        volume, final_stats = select_probability_volume(
            probability_candidates,
            candidate_steps=candidate_steps,
            refine_steps=refine_steps,
            num_phases=options.num_phases,
            target_fraction=target_fraction,
            phase_fraction_tolerance=options.phase_fraction_tolerance,
            anchors=volume_anchors,
            references=references,
            quality=options.quality,
            candidate_deltas=candidate_deltas,
        )
        stats.update(final_stats)
        if "quality_passed" in final_stats and not bool(final_stats["quality_passed"]):
            warnings.warn(
                "prediction returned the least-violation candidate; "
                "inspect quality_* statistics.",
                RuntimeWarning,
                stacklevel=2,
            )
        return quantize_phase(volume, options.num_phases), stats

    def _predict_base(
        self,
        options: PredictOptions,
        *,
        anchors: Sequence[AnchorSlice] | None,
        volume_anchors: Sequence[VolumeAnchor],
        target_labels: torch.Tensor | None,
        t_max: int | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
        anchor_labels = (
            torch.stack([anchor.image for anchor in volume_anchors])
            if volume_anchors
            else None
        )
        anchor_latent, anchor_mask = (
            encode_anchors(
                self.vae,
                anchors,
                num_phases=options.num_phases,
                segment=options.segment_anchors,
                device=self.device,
                peak_strength=options.prior.anchor_strength,
            )
            if anchors and options.prior.anchor_strength > 0.0
            else (None, None)
        )
        latent = sample_latent(
            self.sampler,
            self.vae,
            anchor_latent=anchor_latent,
            anchor_mask=anchor_mask,
            progress=options.progress,
        ).to(self.device)
        guidance_labels = target_labels
        if guidance_labels is None and uses_descriptor_targets(options):
            guidance_labels = anchor_labels
        stats: dict[str, torch.Tensor | int] = {
            "critic_enabled": torch.tensor(
                self.critic is not None and options.critic.weight > 0.0,
                device=self.device,
            )
        }

        candidates, joint_stats = self._run_joint(
            latent,
            options=options,
            anchors=anchors,
            target_labels=guidance_labels,
            critic=self.critic,
            t_max=(int(self.ddpm.num_timesteps) if t_max is None else t_max),
        )
        stats.update(joint_stats)
        target_fraction = self._resolve_fraction(options, guidance_labels)
        reference_labels = target_labels if target_labels is not None else anchor_labels
        reference_probabilities = (
            None
            if reference_labels is None
            else F.one_hot(
                reference_labels.long(),
                num_classes=options.num_phases,
            )
            .movedim(-1, 1)
            .float()
        )
        volume, final_stats = select_latent_volume(
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
        if "quality_passed" in final_stats and not bool(final_stats["quality_passed"]):
            warnings.warn(
                "prediction returned the least-violation candidate; "
                "inspect quality_* statistics.",
                RuntimeWarning,
                stacklevel=2,
            )
        return quantize_phase(volume, options.num_phases), stats

    def _resolve_fraction(
        self,
        options: PredictOptions,
        target_labels: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if options.phase_fractions is not None:
            return torch.tensor(
                options.phase_fractions,
                device=self.device,
                dtype=torch.float32,
            )
        needs_fraction = (
            options.targets.slice_fraction_weight > 0.0
            or options.targets.global_fraction_weight > 0.0
        )
        if not needs_fraction or target_labels is None:
            return None
        labels = target_labels.to(self.device).long()
        return (
            F.one_hot(
                labels,
                num_classes=options.num_phases,
            )
            .float()
            .mean(dim=(0, 1, 2))
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
            critic_weight=options.critic.weight if critic is not None else 0.0,
            anchor_weight=options.joint.anchor_weight if anchors else 0.0,
            fraction_targets=targets.get("fraction_targets"),
            slice_fraction_weight=options.targets.slice_fraction_weight,
            global_fraction_weight=options.targets.global_fraction_weight,
            tpc_targets=targets.get("tpc_targets"),
            tpc_weight=options.targets.tpc_weight,
            sa_targets=targets.get("sa_targets"),
            sa_weight=options.targets.surface_area_weight,
            diffusivity_targets=targets.get("diffusivity_targets"),
            diffusivity_solver=solver,
            diffusivity_weight=options.targets.diffusivity_weight,
            axis_weight=options.joint.axis_weight,
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

        sample_batch = options.scale.decode_batch_size or latent_size * latent_size
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
            batch_size=sample_batch,
            anchor_latent=anchor_latent,
            anchor_mask=anchor_mask,
            progress=options.progress,
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
        return latent, stats

    def _refine_large(
        self,
        latent: torch.Tensor,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
        target_labels: torch.Tensor | None,
        t_max: int,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        targets = self._build_targets(options, target_labels)
        solver = targets.get("diffusivity_solver")

        return optimize_large_latent(
            latent,
            self.vae,
            self.diffusion_model,
            self.ddpm,
            steps=options.scale.steps,
            batch_size=options.scale.batch_size,
            lr=options.scale.learning_rate,
            t_min=options.prior.t_min,
            t_max=t_max,
            num_phases=options.num_phases,
            anchors=anchors,
            segment_anchors=options.segment_anchors,
            sds_weight=options.prior.weight,
            critic=self.critic,
            critic_weight=(
                options.critic.weight if self.critic is not None else 0.0
            ),
            anchor_weight=options.scale.anchor_weight if anchors else 0.0,
            fraction_targets=targets.get("fraction_targets"),
            slice_fraction_weight=options.targets.slice_fraction_weight,
            global_fraction_weight=options.targets.global_fraction_weight,
            tpc_targets=targets.get("tpc_targets"),
            tpc_weight=options.targets.tpc_weight,
            sa_targets=targets.get("sa_targets"),
            sa_weight=options.targets.surface_area_weight,
            diffusivity_targets=targets.get("diffusivity_targets"),
            diffusivity_solver=solver,
            diffusivity_weight=options.targets.diffusivity_weight,
            continuity_weight=options.scale.continuity_weight,
            preservation_weight=options.scale.preservation_weight,
            residual_scale=options.scale.residual_scale,
            checkpoint_every=options.scale.checkpoint_every,
            decode_batch_size=options.scale.decode_batch_size,
            tile_overlap=_tile_overlap(
                int(self.vae.latent_size),
                options.scale.overlap,
            ),
            progress=options.progress,
        )

    def _build_targets(
        self,
        options: PredictOptions,
        target_labels: torch.Tensor | None,
    ) -> DescriptorTargets:
        targets: DescriptorTargets = {}
        if uses_descriptor_targets(options):
            targets.update(
                build_descriptor_targets(
                    target_labels,
                    num_phases=options.num_phases,
                    use_fraction=(
                        (
                            options.targets.slice_fraction_weight > 0.0
                            or options.targets.global_fraction_weight > 0.0
                        )
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
            targets["fraction_targets"] = torch.tensor(
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
