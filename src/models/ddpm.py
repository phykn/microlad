import torch


SCHEDULE_TENSORS = (
    "betas",
    "alphas",
    "alphas_cumprod",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "posterior_variance",
)


class DDPM:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
    ) -> None:
        if timesteps <= 0:
            raise ValueError("timesteps must be positive.")

        if beta_start <= 0.0:
            raise ValueError("beta_start must be positive.")

        if beta_end >= 1.0:
            raise ValueError("beta_end must be smaller than 1.")

        if beta_start >= beta_end:
            raise ValueError("beta_start must be smaller than beta_end.")

        self.device = torch.device(device)
        self.num_timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0
        )
        self.posterior_variance = (
            self.betas * (1.0 - prev) / (1.0 - self.alphas_cumprod)
        )

    def sample_timesteps(
        self,
        batch_size: int,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        device = self.device if device is None else torch.device(device)
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def add_noise(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.q_sample(x_start, t, noise=noise)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._match_device(x_start.device)
        self._validate_timesteps(t, x_start.shape[0])

        if noise is None:
            noise = torch.randn_like(x_start)

        if noise.shape != x_start.shape:
            raise ValueError("noise must have the same shape as x_start.")

        alpha = self._expand(self.sqrt_alphas_cumprod, t, x_start.ndim)
        sigma = self._expand(self.sqrt_one_minus_alphas_cumprod, t, x_start.ndim)
        return alpha * x_start + sigma * noise

    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        self._match_device(x_t.device)
        self._validate_timesteps(t, x_t.shape[0])

        pred_noise = model(x_t, t)
        if pred_noise.shape != x_t.shape:
            raise ValueError("model output must have the same shape as x_t.")

        alpha = self._expand(self.alphas, t, x_t.ndim)
        beta = self._expand(self.betas, t, x_t.ndim)
        sigma = self._expand(self.sqrt_one_minus_alphas_cumprod, t, x_t.ndim)
        mean = (x_t - beta / sigma * pred_noise) / torch.sqrt(alpha)

        noise = torch.randn_like(x_t)
        shape = (t.shape[0],) + (1,) * (x_t.ndim - 1)
        noise = torch.where(t.view(shape) > 0, noise, torch.zeros_like(noise))
        variance = self._expand(self.posterior_variance, t, x_t.ndim)
        return mean + torch.sqrt(variance) * noise

    def _match_device(self, device: torch.device) -> None:
        if self.device == device:
            return

        self.device = device

        for name in SCHEDULE_TENSORS:
            setattr(self, name, getattr(self, name).to(device))

    def _expand(self, values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (ndim - 1)
        return values[t].view(shape)

    def _validate_timesteps(self, t: torch.Tensor, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [B].")

        if t.dtype != torch.long:
            raise ValueError("timesteps must be integer tensors.")

        if t.min().item() < 0 or t.max().item() >= self.num_timesteps:
            raise ValueError("timestep values must be within the DDPM schedule.")
