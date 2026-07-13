import torch
import torch.nn.functional as F


class JointGenerator(torch.nn.Module):
    def __init__(self, volume: torch.Tensor, *, num_phases: int) -> None:
        super().__init__()
        size = int(volume.shape[0])
        coarse_size = max(2, size // 4)
        labels = volume.round().clamp(0, num_phases - 1).to(torch.long)
        one_hot = F.one_hot(labels, num_classes=num_phases).permute(3, 0, 1, 2)
        probabilities = one_hot.to(volume.dtype) * 0.65 + 0.35 / num_phases
        coarse = F.interpolate(
            probabilities.unsqueeze(0),
            size=(coarse_size, coarse_size, coarse_size),
            mode="trilinear",
            align_corners=False,
        )
        noise = torch.randn(
            1,
            8,
            coarse_size,
            coarse_size,
            coarse_size,
            device=volume.device,
            dtype=volume.dtype,
        )
        self.register_buffer("condition", torch.cat([coarse, noise], dim=1))
        self.register_buffer(
            "base_logits",
            probabilities.clamp_min(torch.finfo(volume.dtype).tiny).log().unsqueeze(0),
        )
        self.output_size = size
        self.conv16 = _block(num_phases + 8, 48)
        self.conv32 = _block(48, 32)
        self.conv64 = _block(32, 16)
        self.to_logits = torch.nn.Conv3d(16, num_phases, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.to_logits.weight)
        torch.nn.init.zeros_(self.to_logits.bias)

    def forward(self) -> torch.Tensor:
        values = self.conv16(self.condition)
        middle_size = min(self.output_size, int(values.shape[-1]) * 2)
        values = F.interpolate(
            values,
            size=(middle_size, middle_size, middle_size),
            mode="trilinear",
            align_corners=False,
        )
        values = self.conv32(values)
        values = F.interpolate(
            values,
            size=(self.output_size, self.output_size, self.output_size),
            mode="trilinear",
            align_corners=False,
        )
        return self.base_logits + self.to_logits(self.conv64(values))


class PatchDiscriminator(torch.nn.Module):
    def __init__(self, num_phases: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            _spectral_conv(num_phases, 32),
            torch.nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(32, 64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(64, 128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(128, 1, kernel_size=1)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def build_patch_training(
    real: torch.Tensor | None,
    *,
    num_phases: int,
    device: torch.device,
    dtype: torch.dtype,
    lr: float,
    enabled: bool,
) -> tuple[PatchDiscriminator | None, torch.optim.Optimizer | None, torch.Tensor | None]:
    if not enabled:
        return None, None, None
    if real is None:
        raise ValueError("reference labels are required for patch guidance.")
    discriminator = PatchDiscriminator(num_phases).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
    return discriminator, optimizer, real


def _block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.GroupNorm(8, out_channels),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.GroupNorm(8, out_channels),
        torch.nn.LeakyReLU(0.2, inplace=True),
    )


def _spectral_conv(in_channels: int, out_channels: int) -> torch.nn.Module:
    return torch.nn.utils.spectral_norm(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    )
