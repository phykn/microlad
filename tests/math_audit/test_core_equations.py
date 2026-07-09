import torch

from src.modeling.phases import logits_to_relaxed_labels, phase_logits
from src.modeling.vae import kl_divergence
from src.modeling.diffusion import DDPMProcess
from src.modeling.phases.relaxation import soft_phase_probability


class FixedNoise(torch.nn.Module):
    def __init__(self, noise: torch.Tensor) -> None:
        super().__init__()
        self.noise = noise

    def forward(
        self,
        value: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.noise.expand_as(value)


def test_kl_matches_closed_form_average():
    mu = torch.tensor([[[[0.0]], [[1.0]]]], dtype=torch.float64)
    logvar = torch.zeros_like(mu)

    actual = kl_divergence(mu, logvar)

    assert torch.allclose(actual, torch.tensor(0.25, dtype=torch.float64))


def test_current_kl_reduction_is_invariant_to_repeated_latent_dimensions():
    mu = torch.tensor([[[[1.0]]]], dtype=torch.float64)
    logvar = torch.zeros_like(mu)
    repeated_mu = mu.repeat(1, 4, 8, 8)
    repeated_logvar = logvar.repeat(1, 4, 8, 8)

    scalar_kl = kl_divergence(mu, logvar)
    repeated_kl = kl_divergence(repeated_mu, repeated_logvar)

    assert torch.allclose(repeated_kl, scalar_kl)


def test_distance_logits_decode_symmetrically_between_adjacent_phases():
    value = torch.tensor([[[[0.5]]]], dtype=torch.float64)

    logits = phase_logits(value, num_phases=2, temperature=0.1)
    decoded = logits_to_relaxed_labels(logits, num_phases=2)

    assert torch.allclose(decoded, value)


def test_scalar_phase_expectation_can_turn_bimodal_uncertainty_into_other_phase():
    logits = torch.tensor([[[[0.0]], [[-100.0]], [[0.0]]]])
    categorical = torch.softmax(logits, dim=1)[0, :, 0, 0]

    scalar_value = logits_to_relaxed_labels(logits, num_phases=3)
    recovered = soft_phase_probability(
        scalar_value.reshape(-1),
        num_phases=3,
        temperature=0.1,
        phase_dim=0,
    )[:, 0]

    assert categorical[1] < 1e-20
    assert recovered[1] > 0.99


def test_q_sample_matches_closed_form():
    ddpm = DDPMProcess(timesteps=4, beta_start=0.1, beta_end=0.2)
    clean = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[-0.5]]]])
    timestep = torch.tensor([2], dtype=torch.long)
    expected = (
        ddpm.sqrt_alphas_cumprod[2] * clean
        + ddpm.sqrt_one_minus_alphas_cumprod[2] * noise
    )

    actual = ddpm.q_sample(clean, timestep, noise)

    assert torch.allclose(actual, expected)


def test_p_mean_matches_epsilon_parameterization():
    ddpm = DDPMProcess(timesteps=4, beta_start=0.1, beta_end=0.2)
    noisy = torch.tensor([[[[0.75]]]])
    predicted_noise = torch.tensor([[[[-0.25]]]])
    timestep = torch.tensor([2], dtype=torch.long)
    expected = (
        noisy
        - ddpm.betas[2]
        / ddpm.sqrt_one_minus_alphas_cumprod[2]
        * predicted_noise
    ) / torch.sqrt(ddpm.alphas[2])

    actual = ddpm.p_mean(FixedNoise(predicted_noise), noisy, timestep)

    assert torch.allclose(actual, expected)
