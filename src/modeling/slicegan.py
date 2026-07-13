import torch
import torch.nn.functional as F


SLICEGAN_LATENT_CHANNELS = 32
SLICEGAN_BASE_NOISE_SIZE = 4


def slicegan_output_size(noise_size: int) -> int:
    if (
        not isinstance(noise_size, int)
        or isinstance(noise_size, bool)
        or noise_size <= 0
    ):
        raise ValueError("noise_size must be a positive integer.")
    return noise_size * 16


class SliceGANGenerator(torch.nn.Module):
    def __init__(
        self,
        num_phases: int,
        *,
        fully_convolutional: bool = False,
    ) -> None:
        super().__init__()
        self.fully_convolutional = bool(fully_convolutional)
        channels = (SLICEGAN_LATENT_CHANNELS, 1024, 512, 128, 32)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose3d(
                        source,
                        target,
                        kernel_size=4,
                        stride=2,
                        padding=1 if self.fully_convolutional else 2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm3d(target),
                    torch.nn.ReLU(inplace=True),
                )
                for source, target in zip(channels, channels[1:])
            ]
        )
        self.to_logits = torch.nn.Conv3d(
            channels[-1],
            num_phases,
            kernel_size=3,
            padding=1 if self.fully_convolutional else 0,
            bias=False,
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = noise
        for block in self.blocks:
            x = block(x)
        if not self.fully_convolutional:
            output_shape = tuple(
                slicegan_output_size(int(size)) + 2 for size in noise.shape[-3:]
            )
            x = F.interpolate(
                x,
                size=output_shape,
                mode="trilinear",
                align_corners=False,
            )
        return torch.softmax(self.to_logits(x), dim=1)


class SliceGANCritic(torch.nn.Module):
    def __init__(self, num_phases: int) -> None:
        super().__init__()
        channels = (num_phases, 64, 128, 256, 512, 1)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    source,
                    target,
                    kernel_size=4,
                    stride=2,
                    padding=1 if index < 4 else 0,
                    bias=False,
                )
                for index, (source, target) in enumerate(zip(channels, channels[1:]))
            ]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
