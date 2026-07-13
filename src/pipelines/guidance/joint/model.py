import torch


class LatentRefiner(torch.nn.Module):
    def __init__(
        self,
        latent_ch: int,
        *,
        scale: float,
        hidden_ch: int = 32,
    ) -> None:
        super().__init__()
        if latent_ch <= 0:
            raise ValueError("latent_ch must be positive.")
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        if hidden_ch <= 0 or hidden_ch % 8 != 0:
            raise ValueError("hidden_ch must be a positive multiple of 8.")

        self.scale = float(scale)
        self.body = torch.nn.Sequential(
            _block(latent_ch, hidden_ch),
            _block(hidden_ch, hidden_ch),
        )
        self.to_residual = torch.nn.Conv3d(
            hidden_ch,
            latent_ch,
            kernel_size=3,
            padding=1,
        )
        torch.nn.init.zeros_(self.to_residual.weight)
        torch.nn.init.zeros_(self.to_residual.bias)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        if base.ndim != 5:
            raise ValueError("base latent must have shape [B, C, D, H, W].")
        residual = torch.tanh(self.to_residual(self.body(base)))
        return base + self.scale * residual


def _block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.GroupNorm(8, out_channels),
        torch.nn.SiLU(),
    )
