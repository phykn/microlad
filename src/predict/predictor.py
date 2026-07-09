from collections.abc import Sequence

import numpy as np
import torch

from src.phases.quantization import quantize_phase
from src.predict.refine import three_axis_refinement
from src.diffusion import DiffusionSampler
from src.predict.anchor import validate_anchors
from src.predict.anchor.latent import prepare_anchor_latents
from src.predict.scale import (
    decode_large_latent_volume,
    refine_large_volume,
    optimize_large_volume,
    sample_large_lmpdd,
)
from src.predict.scale.condition import (
    center_start,
    prepare_scale_anchor_latents,
    prepare_scale_anchor_targets,
    shifted_anchor_slices,
)
from src.predict.sds import optimize_volume
from src.predict.targets import build_sds_targets
from src.predict.types import AnchorSlice, PredictOptions
from src.predict.volume import generate_initial_volume


class Predictor:
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

    def _predict_volume_size(
        self,
        *,
        anchors: Sequence[AnchorSlice] | None,
        volume_size: int | None,
    ) -> int:
        anchor_size = self._anchor_volume_size(anchors)

        if volume_size is None:
            return anchor_size if anchor_size is not None else self._image_size()

        _validate_integer("volume_size", volume_size)

        if volume_size <= 0:
            raise ValueError("volume_size must be positive.")

        if anchor_size is not None and anchor_size not in (self._image_size(), volume_size):
            raise ValueError("anchor image size must match vae.image_size or volume_size.")

        return volume_size

    def _anchor_volume_size(
        self,
        anchors: Sequence[AnchorSlice] | None,
    ) -> int | None:
        if not anchors:
            return None

        size = None

        for anchor in anchors:
            if not isinstance(anchor.image, np.ndarray):
                raise TypeError("anchor image must be a numpy array.")

            if anchor.image.ndim != 2:
                raise ValueError("anchor image must be 2D.")

            height, width = anchor.image.shape

            if height != width:
                raise ValueError("anchor image must be square.")

            if size is None:
                size = int(height)
            elif size != int(height):
                raise ValueError("anchor images must have the same size.")

        return size

    def _validate_anchors(
        self,
        anchors: Sequence[AnchorSlice] | None,
        volume_size: int,
    ) -> None:
        if not anchors:
            return

        image_size = self._image_size()
        anchor_size = self._anchor_volume_size(anchors)

        if anchor_size == image_size and volume_size > image_size:
            validate_anchors(anchors, (image_size, image_size, image_size))
            return

        validate_anchors(anchors, (volume_size, volume_size, volume_size))

    def _scale_anchor_latents(
        self,
        anchors: Sequence[AnchorSlice] | None,
        *,
        options: PredictOptions,
        volume_size: int,
        tile_overlap: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        anchor_size = self._anchor_volume_size(anchors)

        if not anchors or anchor_size not in (self._image_size(), int(volume_size)):
            return None, None

        return prepare_scale_anchor_latents(
            self.vae,
            anchors,
            volume_size=volume_size,
            num_phases=options.num_phases,
            segment=options.anchor_segment,
            device=self.device,
            tile_overlap=tile_overlap,
        )

    def _uses_scale_anchor(
        self,
        anchors: Sequence[AnchorSlice] | None,
        volume_size: int,
    ) -> bool:
        return (
            bool(anchors)
            and volume_size > self._image_size()
            and self._anchor_volume_size(anchors) == self._image_size()
        )

    def _scale_anchor_schedule(
        self,
        anchors: Sequence[AnchorSlice] | None,
        *,
        steps: int,
        batch_size: int = 1,
        volume_size: int,
    ) -> list[tuple[int, int]] | None:
        shifted = shifted_anchor_slices(
            anchors,
            volume_size=volume_size,
            base_size=self._image_size(),
            downsample_factor=self._downsample_factor(),
        )
        if not shifted or steps <= 0:
            return None

        batch_size = int(batch_size)

        if batch_size <= 0:
            raise ValueError("sds_batch_size must be positive.")

        if batch_size > volume_size:
            raise ValueError("sds_batch_size cannot exceed volume_size.")

        remaining = [(int(axis), int(index)) for axis, index in shifted]
        schedule: list[tuple[int, int]] = []

        for _ in range(steps):
            group: list[tuple[int, int]] = []
            used_indices: set[int] = set()

            if remaining:
                axis = remaining[0][0]
                next_remaining: list[tuple[int, int]] = []

                for entry_axis, entry_index in remaining:
                    if (
                        entry_axis == axis
                        and len(group) < batch_size
                        and entry_index not in used_indices
                    ):
                        group.append((entry_axis, entry_index))
                        used_indices.add(entry_index)
                    else:
                        next_remaining.append((entry_axis, entry_index))

                remaining = next_remaining
            else:
                axis = int(torch.randint(0, 3, (), device=self.device).item())

            while len(group) < batch_size:
                index = self._random_unused_index(
                    volume_size,
                    used_indices=used_indices,
                )
                group.append((axis, index))
                used_indices.add(index)

            schedule.extend(group)

        return schedule

    def _random_unused_index(
        self,
        volume_size: int,
        *,
        used_indices: set[int],
    ) -> int:
        for index in torch.randperm(volume_size, device=self.device).tolist():
            index = int(index)

            if index not in used_indices:
                return index

        raise ValueError("sds_batch_size cannot exceed volume_size.")

    def _scale_descriptor_tile_size(
        self,
        options: PredictOptions,
        *,
        target_images: Sequence[np.ndarray] | None,
        volume_size: int,
    ) -> int | None:
        if not self._uses_targets(options):
            return None

        target_size = self._target_image_size(target_images)

        if volume_size == self._image_size():
            if target_size != self._image_size():
                raise ValueError("target images must match vae.image_size.")

            return None

        if target_size == self._image_size():
            return target_size

        if target_size == volume_size:
            return None

        raise ValueError("scale-up target images must match vae.image_size or volume_size.")

    def _target_image_size(
        self,
        target_images: Sequence[np.ndarray] | None,
    ) -> int:
        if not target_images:
            raise ValueError("target_images are required when target losses are enabled.")

        size = None

        for image in target_images:
            if not isinstance(image, np.ndarray):
                raise TypeError("target images must be numpy arrays.")

            if image.ndim != 2:
                raise ValueError("target images must be 2D.")

            height, width = image.shape

            if height != width:
                raise ValueError("scale-up target images must be square.")

            if size is None:
                size = int(height)
            elif size != int(height):
                raise ValueError("target images must have the same shape.")

        return int(size)

    def _scale_latent_size(
        self,
        volume_size: int,
        *,
        factor: int,
        tile_size: int,
    ) -> int:
        volume_size = int(volume_size)

        if volume_size <= 0:
            raise ValueError("volume_size must be positive.")

        if volume_size % factor != 0:
            raise ValueError("volume_size must be divisible by VAE downsample factor.")

        latent_size = volume_size // factor

        if latent_size < tile_size:
            raise ValueError("volume_size must be at least vae.image_size.")

        return latent_size

    def _scale_tile_overlap(
        self,
        tile_size: int,
        tile_overlap: int | None,
    ) -> int:
        if tile_overlap is None:
            return max(tile_size // 4, 1) if tile_size > 1 else 0

        tile_overlap = int(tile_overlap)

        if tile_overlap < 0 or tile_overlap >= tile_size:
            raise ValueError("tile_overlap must be non-negative and smaller than tile_size.")

        return tile_overlap

    def _scale_refine_overlap(self) -> int:
        tile_size = self._image_size()
        return max(tile_size // 4, 1) if tile_size > 1 else 0

    def _sds_t_max(self, options: PredictOptions) -> int:
        t_max = (
            int(self.ddpm.num_timesteps)
            if options.sds_t_max is None
            else int(options.sds_t_max)
        )

        if t_max <= options.sds_t_min:
            raise ValueError("sds_t_max must be greater than sds_t_min.")

        if t_max > int(self.ddpm.num_timesteps):
            raise ValueError("sds_t_max must be at most the DDPMProcess schedule length.")

        return t_max

    def _uses_targets(self, options: PredictOptions) -> bool:
        return (
            options.vf_weight > 0.0
            or options.tpc_weight > 0.0
            or options.sa_weight > 0.0
            or options.diffusivity_weight > 0.0
        )


def _validate_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
