from collections.abc import Sequence

import torch

from ..diffusion import DDPMProcess
from ..model import encode_labels
from .anchor import (
    AnchorSlice,
    build_constraints,
    prepare_anchors,
)
from .options import MPDDOptions
from .sampler import ImageMPDDSampler


class MPDDPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        ddpm: DDPMProcess,
        *,
        image_size: int,
        num_phases: int,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.num_phases = int(num_phases)
        self.sampler = ImageMPDDSampler(
            model,
            ddpm,
            image_size=self.image_size,
            num_phases=self.num_phases,
            device=self.device,
        )

    @torch.no_grad()
    def predict(
        self,
        options: MPDDOptions,
        anchors: Sequence[AnchorSlice] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if not isinstance(options, MPDDOptions):
            raise TypeError("options must be MPDDOptions.")
        if options.num_phases != self.num_phases:
            raise ValueError("options.num_phases must match the trained MPDD model.")
        if options.volume_size < self.image_size:
            raise ValueError("volume_size must be at least the trained image size.")

        prepared = prepare_anchors(
            anchors,
            volume_size=options.volume_size,
            num_phases=self.num_phases,
            segment=options.segment_anchors,
            device=self.device,
        )
        anchor_labels, anchor_mask = build_constraints(
            (options.volume_size,) * 3,
            prepared,
            device=self.device,
        )
        target = (
            None
            if options.phase_fractions is None
            else torch.tensor(
                options.phase_fractions,
                device=self.device,
                dtype=torch.float32,
            )
        )
        has_anchors = bool(anchor_mask.any().item())
        clean_anchor = None
        sample_mask = None
        if has_anchors:
            clean_anchor = encode_labels(
                anchor_labels[None, None],
                self.num_phases,
            )[0]
            sample_mask = anchor_mask[None]

        overlap = (
            0
            if options.tile_overlap == 0.0
            else max(int(self.image_size * options.tile_overlap), 1)
        )
        image = self.sampler.sample(
            options.volume_size,
            phase_fractions=target,
            anchor_image=clean_anchor,
            anchor_mask=sample_mask,
            harmonization_steps=options.harmonization_steps,
            tile_overlap=overlap,
            batch_size=options.batch_size,
            ddim_steps=options.ddim_steps,
            guidance_scale=options.guidance_scale,
            progress=options.progress,
        )
        volume = image.argmax(dim=0).to(torch.uint8).cpu()
        fractions = (
            torch.bincount(
                volume.to(torch.long).flatten(),
                minlength=self.num_phases,
            ).float()
            / volume.numel()
        )
        stats: dict[str, object] = {
            "volume_size": options.volume_size,
            "phase_fractions": tuple(float(value) for value in fractions),
            "anchor_voxels": int(anchor_mask.sum().item()),
            "sampling_steps": (
                self.sampler.ddpm.num_timesteps
                if options.ddim_steps is None
                else options.ddim_steps
            ),
            "sampler": "ddpm" if options.ddim_steps is None else "ddim",
            "guidance_scale": options.guidance_scale,
        }
        return volume, stats
