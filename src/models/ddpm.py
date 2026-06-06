import torch


class DDPM:
    """DDPM scheduler for latent diffusion."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.num_timesteps = timesteps

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_om_acp = torch.sqrt(1.0 - self.alphas_cumprod)

        prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = self.betas * (1.0 - prev) / (1.0 - self.alphas_cumprod)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        b = t.shape[0]
        return (
            self.sqrt_acp[t].view(b, 1, 1, 1) * x_start
            + self.sqrt_om_acp[t].view(b, 1, 1, 1) * noise
        )

    def p_sample(self, model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = t.shape[0]
        coef1 = 1.0 / torch.sqrt(self.alphas[t]).view(b, 1, 1, 1)
        coef2 = self.betas[t].view(b, 1, 1, 1) / self.sqrt_om_acp[t].view(b, 1, 1, 1)

        pred = model(x_t, t)
        mean = coef1 * (x_t - coef2 * pred)
        noise = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)
        var = self.posterior_variance[t].view(b, 1, 1, 1)
        return mean + torch.sqrt(var) * noise
