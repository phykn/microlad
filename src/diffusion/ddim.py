import torch

from ..misc import require_int
from .ddpm import DDPMProcess


class DDIMProcess:
    def __init__(self, ddpm: DDPMProcess, sampling_steps: int) -> None:
        if not isinstance(ddpm, DDPMProcess):
            raise TypeError("ddpm must be a DDPMProcess.")
        require_int("sampling_steps", sampling_steps)
        if sampling_steps <= 0 or sampling_steps > ddpm.num_timesteps:
            raise ValueError(
                "sampling_steps must be between 1 and the DDPM timestep count."
            )

        self.ddpm = ddpm
        self.num_timesteps = ddpm.num_timesteps
        self.sampling_steps = sampling_steps
        self.schedule = self._make_schedule()

    def step(
        self,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        *,
        step: int,
        prev_step: int,
    ) -> torch.Tensor:
        if pred_noise.shape != x_t.shape:
            raise ValueError("pred_noise must have the same shape as x_t.")

        alpha = self._get_alpha(step, reference=x_t)
        prev_alpha = self._get_alpha(prev_step, reference=x_t)
        start = (x_t - torch.sqrt(1.0 - alpha) * pred_noise) / torch.sqrt(alpha)
        return (
            torch.sqrt(prev_alpha) * start + torch.sqrt(1.0 - prev_alpha) * pred_noise
        )

    def renoise(
        self,
        x_source: torch.Tensor,
        *,
        source_step: int,
        target_step: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        require_int("source_step", source_step)
        require_int("target_step", target_step)
        if target_step <= source_step:
            raise ValueError("target_step must be noisier than source_step.")

        source_alpha = self._get_alpha(source_step, reference=x_source)
        target_alpha = self._get_alpha(target_step, reference=x_source)
        if noise is None:
            noise = torch.randn_like(x_source)
        if noise.shape != x_source.shape:
            raise ValueError("noise must have the same shape as x_source.")

        ratio = target_alpha / source_alpha
        return torch.sqrt(ratio) * x_source + torch.sqrt(1.0 - ratio) * noise

    def _make_schedule(self) -> list[tuple[int, int]]:
        selected = (
            [self.num_timesteps - 1]
            if self.sampling_steps == 1
            else (
                torch.linspace(
                    0,
                    self.num_timesteps - 1,
                    self.sampling_steps,
                )
                .round()
                .to(torch.long)
                .unique()
                .tolist()
            )
        )
        selected.reverse()
        prev = selected[1:] + [-1]
        return list(zip(selected, prev, strict=True))

    def _get_alpha(
        self,
        step: int,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        require_int("step", step)
        if step < -1 or step >= self.num_timesteps:
            raise ValueError("step must be within the DDPM schedule or -1.")
        if step == -1:
            return torch.ones((), device=reference.device, dtype=reference.dtype)
        return self.ddpm.alphas_cumprod[step].to(
            device=reference.device,
            dtype=reference.dtype,
        )
