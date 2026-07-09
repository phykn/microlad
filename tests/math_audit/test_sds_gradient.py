import torch

from src.diffusion import DDPMProcess
from src.predict.sds.core import sds_loss
from tests.math_audit.helpers import cosine_similarity


class ConstantPrediction(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full_like(noisy, self.value)


def test_current_sds_gradient_is_parallel_to_paper_pseudogradient():
    ddpm = DDPMProcess(timesteps=4, beta_start=0.1, beta_end=0.2)
    latent = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
    noise = torch.full_like(latent, 1.25)
    timestep = torch.tensor([2], dtype=torch.long)

    loss, _ = sds_loss(
        latent,
        ConstantPrediction(0.25),
        ddpm,
        t_min=1,
        t_max=3,
        t=timestep,
        noise=noise,
    )
    (current,) = torch.autograd.grad(loss, latent)

    alpha_bar = ddpm.alphas_cumprod[timestep].view(1, 1, 1, 1)
    paper_weight = (1.0 - alpha_bar) / alpha_bar
    paper = (
        2.0
        * paper_weight
        * torch.sqrt(alpha_bar)
        * (0.25 - noise)
        / latent.numel()
    )

    assert torch.allclose(
        cosine_similarity(current, paper),
        torch.tensor(1.0),
    )
    expected_ratio = torch.sqrt(alpha_bar) / 2.0
    assert torch.allclose(current / paper, expected_ratio.expand_as(current))
