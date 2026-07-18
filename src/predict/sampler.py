from collections.abc import Sequence

import torch
from tqdm import tqdm

from ..diffusion import DDIMProcess, DDPMProcess
from ..misc import require_int, require_number
from .noise import predict_tiles
from .volume import merge_planes, slice_volume


class ImageMPDDSampler:
    """Builds a 3D categorical field by rotating a trained 2D DDPM over axes."""

    def __init__(
        self,
        model: torch.nn.Module,
        ddpm: DDPMProcess,
        *,
        image_size: int,
        num_phases: int,
        device: str | torch.device,
    ) -> None:
        require_int("image_size", image_size)
        require_int("num_phases", num_phases)
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if num_phases < 2:
            raise ValueError("num_phases must be at least 2.")
        if getattr(ddpm, "num_timesteps", 0) <= 0:
            raise ValueError("ddpm must define a positive num_timesteps.")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.ddpm = ddpm
        self.image_size = int(image_size)
        self.num_phases = int(num_phases)

    @torch.no_grad()
    def sample(
        self,
        volume_size: int,
        *,
        phase_fractions: torch.Tensor | Sequence[float] | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        harmonization_steps: int = 10,
        tile_overlap: int = 0,
        batch_size: int = 8,
        ddim_steps: int | None = None,
        guidance_scale: float = 1.0,
        progress: bool = True,
    ) -> torch.Tensor:
        require_int("volume_size", volume_size)
        require_int("harmonization_steps", harmonization_steps)
        require_int("tile_overlap", tile_overlap)
        require_int("batch_size", batch_size)
        require_number("guidance_scale", guidance_scale)
        if ddim_steps is not None:
            require_int("ddim_steps", ddim_steps)
        if volume_size < self.image_size:
            raise ValueError("volume_size must be at least image_size.")
        if harmonization_steps <= 0:
            raise ValueError("harmonization_steps must be positive.")
        if tile_overlap < 0 or tile_overlap >= self.image_size:
            raise ValueError("tile_overlap must be smaller than image_size.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative.")
        if not isinstance(progress, bool):
            raise ValueError("progress must be a boolean.")
        if (anchor_image is None) != (anchor_mask is None):
            raise ValueError("anchor_image and anchor_mask must be provided together.")

        ddim = None if ddim_steps is None else DDIMProcess(self.ddpm, ddim_steps)
        shape = (self.num_phases, volume_size, volume_size, volume_size)
        volume = torch.randn(shape, device=self.device)
        fractions = self._get_fractions(phase_fractions, dtype=volume.dtype)
        if fractions is None and guidance_scale != 1.0:
            raise ValueError(
                "phase_fractions are required when guidance_scale is not one."
            )
        anchor_image, anchor_mask = self._get_anchor(
            anchor_image,
            anchor_mask,
            shape=shape,
            dtype=volume.dtype,
        )
        overlap = 0 if volume_size == self.image_size else tile_overlap

        self.model.eval()
        schedule = (
            [(step, step - 1) for step in range(self.ddpm.num_timesteps - 1, -1, -1)]
            if ddim is None
            else ddim.schedule
        )
        bar = tqdm(
            schedule,
            total=len(schedule),
            desc="Image MPDD",
            disable=not progress,
        )
        for index, (step, prev_step) in enumerate(bar):
            axis = index % 3
            planes = slice_volume(volume, axis)
            steps = torch.full(
                (planes.shape[0],),
                step,
                dtype=torch.long,
                device=self.device,
            )
            plane_anchor = None
            plane_mask = None
            if (
                anchor_image is not None
                and anchor_mask is not None
            ):
                plane_anchor = slice_volume(anchor_image, axis)
                plane_mask = slice_volume(anchor_mask, axis)
            for repeat in range(harmonization_steps):
                noise = predict_tiles(
                    self.model,
                    planes,
                    steps,
                    tile_size=self.image_size,
                    overlap=overlap,
                    batch_size=batch_size,
                    fractions=fractions,
                    axis_condition=axis,
                    guidance=guidance_scale,
                    anchor_image=plane_anchor,
                    anchor_mask=plane_mask,
                )
                planes = (
                    self.ddpm.sample_step(
                        planes,
                        steps,
                        noise,
                    )
                    if ddim is None
                    else ddim.step(
                        planes,
                        noise,
                        step=step,
                        prev_step=prev_step,
                    )
                )
                volume = merge_planes(planes, axis)
                if repeat + 1 >= harmonization_steps or step == 0:
                    break
                planes = slice_volume(volume, axis)
                if ddim is None:
                    planes = self.ddpm.renoise(planes, steps)
                else:
                    planes = ddim.renoise(
                        planes,
                        source_step=prev_step,
                        target_step=step,
                    )

        if not torch.isfinite(volume).all():
            raise ValueError("MPDD sampling produced non-finite values.")
        return volume

    def _get_fractions(
        self,
        phase_fractions: torch.Tensor | Sequence[float] | None,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if phase_fractions is None:
            return None
        fractions = torch.as_tensor(
            phase_fractions,
            device=self.device,
            dtype=dtype,
        )
        if fractions.shape != (self.num_phases,):
            raise ValueError("phase_fractions must have shape [num_phases].")
        if not torch.isfinite(fractions).all() or torch.any(fractions < 0.0):
            raise ValueError("phase_fractions must be finite and non-negative.")
        if not torch.allclose(
            fractions.sum(), torch.ones((), device=self.device), atol=1e-4
        ):
            raise ValueError("phase_fractions must sum to one.")
        return fractions

    def _get_anchor(
        self,
        anchor_image: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
        *,
        shape: tuple[int, int, int, int],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (anchor_image is None) != (anchor_mask is None):
            raise ValueError("anchor_image and anchor_mask must be provided together.")
        if anchor_image is None or anchor_mask is None:
            return None, None
        image = anchor_image.to(device=self.device, dtype=dtype)
        if image.shape != torch.Size(shape) or not torch.isfinite(image).all():
            raise ValueError(
                "anchor_image must be finite and match the sampled volume."
            )
        mask = anchor_mask.to(device=self.device, dtype=torch.bool)
        if mask.shape != torch.Size((1, *shape[1:])):
            raise ValueError("anchor_mask must have shape [1, D, H, W].")
        return image, mask
