from collections.abc import Sequence

import numpy as np
import torch

from src.app.api.options import AnchorSlice, PredictOptions
from src.common.validation import require_int
from src.modeling.vae import get_downsample_factor
from src.pipelines.guidance.conditioning.validation import validate_anchors
from src.pipelines.scaling.conditioning import (
    center_start,
    prepare_scale_anchor_latents,
    shifted_anchor_slices,
)

class PredictionPreparation:
    def _predict_volume_size(
        self,
        *,
        anchors: Sequence[AnchorSlice] | None,
        volume_size: int | None,
    ) -> int:
        anchor_size = self._anchor_volume_size(anchors)

        if volume_size is None:
            return anchor_size if anchor_size is not None else self._image_size()

        require_int("volume_size", volume_size)

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
            downsample_factor=get_downsample_factor(self.vae),
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
