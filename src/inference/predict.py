from dataclasses import dataclass

import torch

from .conditions import (
    ConditionLock,
    EncodedConditions,
    FixedSlice,
    ParsedConditionImage,
    encode_condition_items,
    encode_tiled_condition_items,
    parse_condition_options,
)
from .condition_stats import build_condition_stats
from .locked_sampling import sample_locked_latent_volume
from .decoding import multi_axis_decode, three_axis_refinement
from .sds import sds_refine_volume


@dataclass
class GenerationOptions:
    volume_shape: tuple[int, int, int, int] = (4, 16, 16, 16)
    refinement_steps: int = 0
    sds_steps: int = 0
    sds_unet: torch.nn.Module | None = None
    sds_lr: float = 1e-2
    t_min: int = 50
    t_max: int = 950
    lock_condition_slice: bool = True
    lock_strength: float = 1.0


class MicroLadPredictor:
    def __init__(
        self,
        vae: torch.nn.Module,
        unet: torch.nn.Module,
        ddpm,
        options: GenerationOptions | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.vae = vae
        self.unet = unet
        self.ddpm = ddpm
        self.options = options or GenerationOptions()
        self.device = torch.device(device)

    def _default_size(self, downsample: int = 4) -> int:
        return int(self.options.volume_shape[1]) * downsample

    def _encode_conditions(
        self,
        conditions: list[ParsedConditionImage],
        size: int,
        downsample: int,
    ) -> tuple[EncodedConditions, int | None, int]:
        options = self.options
        if size <= 64:
            return (
                encode_condition_items(
                    vae=self.vae,
                    conditions=conditions,
                    lock_condition_slice=options.lock_condition_slice,
                    device=self.device,
                    image_size=size,
                ),
                None,
                0,
            )

        if not conditions:
            return (
                EncodedConditions(locks=[], fixed_slices=[], condition_slices=[], condition_images=[]),
                64 // downsample,
                16 // downsample,
            )

        return (
            encode_tiled_condition_items(
                vae=self.vae,
                conditions=conditions,
                tile_size=64,
                tile_overlap=16,
                downsample=downsample,
                lock_condition_slice=options.lock_condition_slice,
                device=self.device,
                image_size=size,
            ),
            64 // downsample,
            16 // downsample,
        )

    def _refine_decoded_volume(
        self,
        volume: torch.Tensor,
        condition_images: list[torch.Tensor] | None,
        condition_slices: list[FixedSlice] | None,
        condition_weight: float,
        stats_weight: float,
        fixed_slices: list[FixedSlice] | None,
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        options = self.options
        if options.refinement_steps > 0:
            volume = three_axis_refinement(volume, self.vae, refinement_steps=options.refinement_steps)

        if not fixed_slices:
            fixed_slices = None

        if options.sds_steps <= 0:
            if condition_weight > 0 or stats_weight > 0:
                raise ValueError("condition_weight and stats_weight require sds_steps > 0.")
            if fixed_slices:
                volume, _ = sds_refine_volume(
                    volume=volume,
                    vae=self.vae,
                    unet=self.unet,
                    ddpm=self.ddpm,
                    steps=0,
                    lr=options.sds_lr,
                    t_min=options.t_min,
                    t_max=options.t_max,
                    fixed_slices=fixed_slices,
                )
            return volume, []

        phases = [0, 1]
        stats = build_condition_stats(
            condition_images=condition_images,
            stats_weight=stats_weight,
            phases=phases,
            device=self.device,
        )

        denoise_unet = options.sds_unet if options.sds_unet is not None else self.unet
        return sds_refine_volume(
            volume=volume,
            vae=self.vae,
            unet=denoise_unet,
            ddpm=self.ddpm,
            steps=options.sds_steps,
            lr=options.sds_lr,
            t_min=options.t_min,
            t_max=options.t_max,
            refinement_steps=options.refinement_steps,
            phases=phases,
            vf_moments=stats.vf_moments,
            vf_weight=stats_weight,
            grayscale_tpc_target=stats.grayscale_tpc_target,
            grayscale_tpc_bin_mat=stats.grayscale_tpc_bin_mat,
            grayscale_tpc_bin_counts=stats.grayscale_tpc_bin_counts,
            grayscale_tpc_weight=stats_weight,
            sa_targets=stats.sa_targets,
            sa_weight=stats_weight,
            condition_slices=condition_slices,
            condition_weight=condition_weight,
            fixed_slices=fixed_slices,
        )

    def _sample_and_refine(
        self,
        locks: list[ConditionLock],
        volume_shape: tuple[int, int, int, int],
        condition_images: list[torch.Tensor] | None,
        condition_slices: list[FixedSlice] | None,
        condition_weight: float,
        stats_weight: float,
        fixed_slices: list[FixedSlice] | None,
        tile_size: int | None = None,
        tile_overlap: int = 0,
        downsample: int = 4,
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        options = self.options
        self.vae.eval()
        self.unet.eval()
        volume_z = sample_locked_latent_volume(
            unet=self.unet,
            ddpm=self.ddpm,
            locks=locks,
            volume_shape=volume_shape,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            lock_strength=options.lock_strength,
            downsample=downsample,
            device=self.device,
        )
        volume = multi_axis_decode(self.vae, volume_z, downsample=downsample)
        volume, sds_history = self._refine_decoded_volume(
            volume=volume,
            condition_images=condition_images,
            condition_slices=condition_slices,
            condition_weight=condition_weight,
            stats_weight=stats_weight,
            fixed_slices=fixed_slices,
        )
        return volume, sds_history

    @torch.no_grad()
    def predict(
        self,
        condition: dict[str, object] | None = None,
    ) -> dict[str, object]:
        options = self.options
        downsample = 4
        parsed = parse_condition_options(
            condition=condition,
            default_size=self._default_size(downsample=downsample),
            downsample=downsample,
        )
        volume_shape = (
            options.volume_shape[0],
            parsed.size // downsample,
            parsed.size // downsample,
            parsed.size // downsample,
        )
        encoded, tile_size, tile_overlap = self._encode_conditions(
            conditions=parsed.images,
            size=parsed.size,
            downsample=downsample,
        )

        volume, sds_history = self._sample_and_refine(
            locks=encoded.locks,
            volume_shape=volume_shape,
            condition_images=encoded.condition_images,
            condition_slices=encoded.condition_slices,
            condition_weight=parsed.condition_weight,
            stats_weight=parsed.stats_weight,
            fixed_slices=encoded.fixed_slices,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            downsample=downsample,
        )

        return {
            "volume": volume,
            "sds_history": sds_history,
        }
