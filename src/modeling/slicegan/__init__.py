from src.modeling.slicegan.model import (
    MIN_SLICE_SIZE as MIN_SLICE_SIZE,
    NOISE_CHANNELS as NOISE_CHANNELS,
    SCALE_FACTOR as SCALE_FACTOR,
    SliceGANCritic as SliceGANCritic,
    SliceGANGenerator as SliceGANGenerator,
    output_size as output_size,
)
from src.modeling.slicegan.objective import (
    critic_loss as critic_loss,
    generator_loss as generator_loss,
    gradient_penalty as gradient_penalty,
)
