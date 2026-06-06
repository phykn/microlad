import torch
import torch.nn as nn

from models import DDPM

from .diffusion import diffusion_noise_loss


class SliceConditionedDiffusionLoss(nn.Module):
    def __init__(
        self,
        vae: torch.nn.Module,
        ddpm: DDPM,
        condition_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be between 0 and 1.")
        self.vae = vae
        self.ddpm = ddpm
        self.condition_dropout = condition_dropout

    def forward(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(model.parameters()).device
        base_model = model.module if hasattr(model, "module") else model
        target = batch["target"].to(device)
        conditions, axes, slice_indices = self._get_conditions(batch, device)

        with torch.no_grad():
            target_z, _ = self.vae.encode(target * 2 - 1)
            b, k, c, h, w = conditions.shape
            flat_conditions = conditions.reshape(b * k, c, h, w)
            condition_z, _ = self.vae.encode(flat_conditions * 2 - 1)
            condition_z = condition_z.reshape(b, k, *condition_z.shape[1:])

        t = torch.randint(0, self.ddpm.num_timesteps, (target_z.shape[0],), device=device)
        noise = torch.randn_like(target_z)
        z_t = self.ddpm.q_sample(target_z, t, noise)

        losses = []
        dropout_rates = []
        for condition_idx in range(condition_z.shape[1]):
            current_condition = condition_z[:, condition_idx]
            current_axis = axes[:, condition_idx]
            current_slice_index = slice_indices[:, condition_idx]

            dropout_mask = torch.rand(target_z.shape[0], device=device) < self.condition_dropout
            if dropout_mask.any():
                current_condition = current_condition.clone()
                current_axis = current_axis.clone()
                current_slice_index = current_slice_index.clone()
                current_condition[dropout_mask] = 0.0
                current_axis[dropout_mask] = base_model.null_axis
                current_slice_index[dropout_mask] = base_model.null_slice

            pred_noise = model(z_t, t, current_condition, current_axis, current_slice_index)
            losses.append(diffusion_noise_loss(pred_noise, noise))
            dropout_rates.append(dropout_mask.float().mean())

        loss = torch.stack(losses).mean()
        dropout_rate = torch.stack(dropout_rates).mean().detach()
        condition_count = torch.tensor(float(condition_z.shape[1]), device=device)
        return {
            "loss": loss.detach(),
            "diffusion": loss.detach(),
            "condition_dropout": dropout_rate,
            "condition_count": condition_count,
        }, loss

    @staticmethod
    def _get_conditions(
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "conditions" in batch:
            conditions = batch["conditions"].to(device)
            axes = batch["axes"].to(device).long()
            slice_indices = batch["slice_indices"].to(device).long()
        else:
            conditions = batch["condition"].to(device).unsqueeze(1)
            axes = batch["axis"].to(device).long().unsqueeze(1)
            slice_indices = batch["slice_index"].to(device).long().unsqueeze(1)

        if conditions.ndim != 5:
            raise ValueError("conditions must have shape [B, K, C, H, W].")
        if axes.shape != slice_indices.shape or axes.shape != conditions.shape[:2]:
            raise ValueError("axes and slice_indices must have shape [B, K].")
        return conditions, axes, slice_indices
