from collections.abc import Sequence

import torch
from tqdm import tqdm

from ..diffusion import DDIMProcess, DDPMProcess
from ..misc import require_int
from ..model import encode_labels
from .anchor import (
    AnchorSlice,
    build_constraints,
    prepare_anchors,
)
from .options import MPDDOptions
from .noise import guide_noise
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
    def predict_images(
        self,
        axis: int,
        count: int,
        *,
        phase_fractions: Sequence[float] | None = None,
        ddim_steps: int | None = None,
        guidance_scale: float = 1.0,
        progress: bool = True,
    ) -> torch.Tensor:
        require_int("axis", axis)
        require_int("count", count)
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2.")
        if count <= 0:
            raise ValueError("count must be positive.")

        opts = MPDDOptions(
            num_phases=self.num_phases,
            phase_fractions=phase_fractions,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            progress=progress,
        )
        x = torch.randn(
            count,
            self.num_phases,
            self.image_size,
            self.image_size,
            device=self.device,
        )
        cond = (
            None
            if opts.phase_fractions is None
            else torch.tensor(
                opts.phase_fractions,
                device=self.device,
                dtype=x.dtype,
            ).expand(count, -1)
        )
        axes = torch.full(
            (count,),
            axis,
            dtype=torch.long,
            device=self.device,
        )
        ddpm = self.sampler.ddpm
        ddim = None if opts.ddim_steps is None else DDIMProcess(ddpm, opts.ddim_steps)
        schedule = (
            [(step, step - 1) for step in range(ddpm.num_timesteps - 1, -1, -1)]
            if ddim is None
            else ddim.schedule
        )
        self.sampler.model.eval()
        bar = tqdm(
            schedule,
            total=len(schedule),
            desc=f"axis {axis}",
            disable=not opts.progress,
        )
        for step, prev in bar:
            steps = torch.full(
                (count,),
                step,
                dtype=torch.long,
                device=self.device,
            )
            noise = guide_noise(
                self.sampler.model,
                x,
                steps,
                condition=cond,
                axis_condition=axes,
                guidance=opts.guidance_scale,
            )
            x = (
                ddpm.sample_step(x, steps, noise)
                if ddim is None
                else ddim.step(x, noise, step=step, prev_step=prev)
            )
        return x.argmax(dim=1).to(torch.uint8).cpu()

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
