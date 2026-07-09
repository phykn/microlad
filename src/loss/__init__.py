from src.loss.diffusion import DiffusionLoss, diffusion_loss
from src.phases import (
    logits_to_relaxed_labels,
    phase_cross_entropy,
    phase_levels,
    phase_logits,
    phase_loss,
    phase_target_indices,
)
from src.vae import VAELoss, kl_divergence, vae_loss
