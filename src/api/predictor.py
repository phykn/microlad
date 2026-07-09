from collections.abc import Sequence

import numpy as np
import torch

from src.phases.quantization import quantize_phase
from src.reconstruction.refinement import three_axis_refinement
from src.diffusion import DiffusionSampler
from src.guidance.conditioning.validation import validate_anchors
from src.guidance.conditioning.latents import prepare_anchor_latents
from src.scaling.decoding import decode_large_latent_volume
from src.scaling.optimization import optimize_large_volume
from src.scaling.refinement import refine_large_volume
from src.scaling.sampling import sample_large_lmpdd
from src.scaling.conditioning import (
    center_start,
    prepare_scale_anchor_latents,
    prepare_scale_anchor_targets,
    shifted_anchor_slices,
)
from src.guidance.optimization import optimize_volume
from src.guidance.conditioning.targets import build_sds_targets
from src.api.options import AnchorSlice, PredictOptions
from src.reconstruction.volume import generate_initial_volume


from src.api.preparation import PredictionPreparation

class Predictor(PredictionPreparation):
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
        volume_size = self._predict_volume_size(
            anchors=anchors,
            volume_size=volume_size,
        )
        self._validate_anchors(anchors, volume_size)
        self._validate_predict_inputs(options, target_images=target_images)

        volume, stats = self._generate_volume(
            volume_size,
            options=options,
            anchors=anchors,
        )

        if options.sds_steps > 0:
            volume, sds_stats = self._run_sds(
                volume,
                options=options,
                anchors=anchors,
                target_images=target_images,
            )

            stats = {**stats, **sds_stats}

        if options.refine_steps > 0:
            volume = self._refine_volume(volume, options.refine_steps)

        return quantize_phase(volume, options.num_phases), stats

    def _generate_volume(
        self,
        volume_size: int,
        *,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
    ) -> tuple[torch.Tensor, dict]:
        if volume_size == self._image_size():
            anchor_latent, anchor_mask = prepare_anchor_latents(
                self.vae,
                anchors,
                num_phases=options.num_phases,
                segment=options.anchor_segment,
                device=self.device,
            )

            volume = generate_initial_volume(
                self.sampler,
                self.vae,
                size=self._image_size(),
                anchor_latent=anchor_latent,
                anchor_mask=anchor_mask,
            ).to(self.device)
            return volume, {}

        return self._generate_large_volume(
            volume_size,
            options=options,
            anchors=anchors,
        )

    def _generate_large_volume(
        self,
        volume_size: int,
        *,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        factor = self._downsample_factor()
        tile_size = int(self.vae.latent_size)
        overlap = self._scale_tile_overlap(tile_size, None)
        latent_size = self._scale_latent_size(
            volume_size,
            factor=factor,
            tile_size=tile_size,
        )
        anchor_latent, anchor_mask = self._scale_anchor_latents(
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
        volume = decode_large_latent_volume(
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
                base_size=self._image_size(),
            ),
        }
        return volume, stats

    def _refine_volume(self, volume: torch.Tensor, steps: int) -> torch.Tensor:
        if volume.shape[0] == self._image_size():
            return three_axis_refinement(
                volume,
                self.vae,
                steps=steps,
            )

        return refine_large_volume(
            volume,
            self.vae,
            steps=steps,
            tile_overlap=self._scale_refine_overlap(),
        )

    def _run_sds(
        self,
        volume: torch.Tensor,
        *,
        options: PredictOptions,
        anchors: Sequence[AnchorSlice] | None,
        target_images: Sequence[np.ndarray] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        kwargs = self._sds_kwargs(
            options,
            anchors=anchors,
            target_images=target_images,
            volume_size=int(volume.shape[0]),
        )

        if volume.shape[0] != self._image_size():
            return optimize_large_volume(
                volume,
                self.vae,
                self.diffusion_model,
                self.ddpm,
                tile_overlap=self._scale_refine_overlap(),
                **kwargs,
            )

        kwargs.pop("anchor_targets", None)
        kwargs.pop("anchor_masks", None)
        kwargs.pop("descriptor_tile_size", None)
        return optimize_volume(
            volume,
            self.vae,
            self.diffusion_model,
            self.ddpm,
            **kwargs,
        )

    def _sds_kwargs(
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

        if self._uses_scale_anchor(anchors, volume_size):
            anchor_targets, anchor_masks = prepare_scale_anchor_targets(
                self.vae,
                anchors,
                volume_size=volume_size,
                base_size=self._image_size(),
                num_phases=options.num_phases,
                segment=options.anchor_segment,
                device=self.device,
                dtype=torch.float32,
                downsample_factor=self._downsample_factor(),
            )

            sds_anchors = None
            slice_schedule = self._scale_anchor_schedule(
                anchors,
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
            "t_max": self._sds_t_max(options),
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
            "descriptor_tile_size": self._scale_descriptor_tile_size(
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

    def _validate_predict_inputs(
        self,
        options: PredictOptions,
        *,
        target_images: Sequence[np.ndarray] | None,
    ) -> None:
        uses_targets = self._uses_targets(options)

        if uses_targets and options.sds_steps <= 0:
            raise ValueError("target losses require sds_steps to be positive.")

        if uses_targets and not target_images:
            raise ValueError("target_images are required when target losses are enabled.")

        if options.sds_steps > 0:
            self._sds_t_max(options)

    def _image_size(self) -> int:
        return int(self.vae.image_size)

    def _downsample_factor(self) -> int:
        factor = int(
            getattr(
                self.vae,
                "downsample_factor",
                int(self.vae.image_size) // int(self.vae.latent_size),
            )
        )

        if factor <= 0:
            raise ValueError("VAE downsample factor must be positive.")

        if int(self.vae.image_size) != int(self.vae.latent_size) * factor:
            raise ValueError(
                "vae.image_size must equal vae.latent_size times downsample factor."
            )

        return factor

