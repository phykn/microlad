import torch
import torch.nn as nn
import torch.nn.functional as F


MIN_IMAGE_SIZE = 16
GLOBAL_FEATURE_SIZE = 4


class ImageCritic(nn.Module):
    """Scores decoded 2D phase-probability images at local and global scales."""

    def __init__(
        self,
        num_phases: int,
        image_size: int,
        *,
        base_ch: int = 64,
    ) -> None:
        super().__init__()
        if num_phases <= 0:
            raise ValueError("num_phases must be positive.")
        if image_size < MIN_IMAGE_SIZE:
            raise ValueError(f"image_size must be at least {MIN_IMAGE_SIZE}.")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive.")

        self.num_phases = num_phases
        self.image_size = image_size
        channels = (num_phases, base_ch, base_ch * 2, base_ch * 4)
        layers = []
        for source, target in zip(channels, channels[1:]):
            layers.extend(
                (
                    nn.Conv2d(
                        source,
                        target,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.features = nn.Sequential(*layers)
        self.local_score = nn.Conv2d(
            channels[-1],
            1,
            kernel_size=3,
            padding=1,
        )
        self.global_score = nn.Linear(
            channels[-1] * GLOBAL_FEATURE_SIZE**2,
            1,
        )

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        features = self.morphology_features(probabilities)[-1]
        local_score = self.local_score(features).mean(dim=(-2, -1))
        global_features = F.adaptive_avg_pool2d(
            features,
            output_size=(GLOBAL_FEATURE_SIZE, GLOBAL_FEATURE_SIZE),
        )
        global_score = self.global_score(global_features.flatten(start_dim=1))
        return 0.5 * (local_score + global_score)

    def morphology_features(
        self,
        probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return multi-scale morphology features without spatial pairing."""
        if probabilities.ndim != 4 or probabilities.shape[1] != self.num_phases:
            raise ValueError(
                "phase probabilities must have shape "
                f"[B, {self.num_phases}, H, W]."
            )
        if min(probabilities.shape[-2:]) < MIN_IMAGE_SIZE:
            raise ValueError(
                f"phase probability image size must be at least {MIN_IMAGE_SIZE}."
            )
        expected_size = (self.image_size, self.image_size)
        if probabilities.shape[-2:] != expected_size:
            raise ValueError(
                "phase probability image size must match the configured VAE "
                f"image size {self.image_size}x{self.image_size}."
            )

        values = probabilities
        features = []
        for layer in self.features:
            values = layer(values)
            if isinstance(layer, nn.LeakyReLU):
                features.append(values)
        return tuple(features)
