from dataclasses import dataclass, field

import torch

from loss import build_grayscale_tpc_target
from .conditions import (
    ConditionSpec,
    condition_error_from_volume,
    condition_specs_to_dicts,
    encode_condition,
    encode_condition_items,
    scale_up_volume_shape,
)
from .conditioned_sampling import (
    sample_conditioned_latent_volume,
    sample_conditioned_latent_volume_multi,
    voxel_to_latent_index,
)
from .decoding import multi_axis_decode, three_axis_refinement
from .sds import sds_refine_volume


@dataclass
class PredictConfig:
    conditions: list[ConditionSpec]
    condition_is_latent: bool = False
    volume_shape: tuple[int, int, int, int] = (4, 16, 16, 16)
    refinement_steps: int = 0
    sds_steps: int = 0
    sds_unet: torch.nn.Module | None = None
    sds_lr: float = 1e-2
    t_min: int = 50
    t_max: int = 950
    use_condition_tpc: bool = False
    condition_tpc_weight: float = 0.0
    lock_condition_slice: bool = True
    device: str | torch.device = "cpu"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ScaleUpConfig:
    conditions: list[ConditionSpec]
    output_size: int | None = None
    latent_ch: int = 4
    downsample: int = 4
    refinement_steps: int = 0
    sds_steps: int = 0
    sds_unet: torch.nn.Module | None = None
    sds_lr: float = 1e-2
    t_min: int = 50
    t_max: int = 950
    use_condition_tpc: bool = False
    condition_tpc_weight: float = 0.0
    lock_condition_slice: bool = True
    device: str | torch.device = "cpu"


def _refine_decoded_volume(
    volume: torch.Tensor,
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    refinement_steps: int,
    sds_steps: int,
    sds_unet: torch.nn.Module | None,
    sds_lr: float,
    t_min: int,
    t_max: int,
    use_condition_tpc: bool,
    condition_tpc_weight: float,
    condition_image: torch.Tensor | None,
    fixed_slices: list[dict[str, torch.Tensor | int]] | None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    if refinement_steps > 0:
        volume = three_axis_refinement(volume, vae, refinement_steps=refinement_steps)

    if not fixed_slices:
        fixed_slices = None

    if sds_steps <= 0:
        if fixed_slices:
            volume, _ = sds_refine_volume(
                volume=volume,
                vae=vae,
                unet=unet,
                ddpm=ddpm,
                steps=0,
                lr=sds_lr,
                t_min=t_min,
                t_max=t_max,
                fixed_slices=fixed_slices,
            )
        return volume, []

    grayscale_tpc_target = None
    grayscale_tpc_bin_mat = None
    grayscale_tpc_bin_counts = None
    if use_condition_tpc:
        if condition_image is None:
            raise ValueError("use_condition_tpc requires image-space condition input.")
        grayscale_tpc_target, grayscale_tpc_bin_mat, grayscale_tpc_bin_counts = build_grayscale_tpc_target(
            condition_image
        )

    denoise_unet = sds_unet if sds_unet is not None else unet
    return sds_refine_volume(
        volume=volume,
        vae=vae,
        unet=denoise_unet,
        ddpm=ddpm,
        steps=sds_steps,
        lr=sds_lr,
        t_min=t_min,
        t_max=t_max,
        refinement_steps=refinement_steps,
        grayscale_tpc_target=grayscale_tpc_target,
        grayscale_tpc_bin_mat=grayscale_tpc_bin_mat,
        grayscale_tpc_bin_counts=grayscale_tpc_bin_counts,
        grayscale_tpc_weight=condition_tpc_weight,
        fixed_slices=fixed_slices,
    )


@torch.no_grad()
def predict(
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    condition: torch.Tensor,
    axis: int,
    slice_index: int,
    condition_is_latent: bool = False,
    volume_shape: tuple[int, int, int, int] = (4, 16, 16, 16),
    refinement_steps: int = 0,
    sds_steps: int = 0,
    sds_unet: torch.nn.Module | None = None,
    sds_lr: float = 1e-2,
    t_min: int = 50,
    t_max: int = 950,
    use_condition_tpc: bool = False,
    condition_tpc_weight: float = 0.0,
    lock_condition_slice: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    device = torch.device(device)
    unet.eval()

    if condition_is_latent:
        condition_z, condition_image = encode_condition(vae, condition, condition_is_latent=True, device=device)
    else:
        vae.eval()
        condition_z, condition_image = encode_condition(vae, condition, condition_is_latent=False, device=device)

    volume_z = sample_conditioned_latent_volume(
        unet=unet,
        ddpm=ddpm,
        condition_z=condition_z,
        axis=axis,
        slice_index=slice_index,
        volume_shape=volume_shape,
        device=device,
    )
    vae.eval()
    volume = multi_axis_decode(vae, volume_z)
    fixed_slices = None
    if lock_condition_slice and condition_image is not None:
        fixed_slices = [{"axis": axis, "index": slice_index, "image": condition_image.squeeze(0)}]
    volume, sds_history = _refine_decoded_volume(
        volume=volume,
        vae=vae,
        unet=unet,
        ddpm=ddpm,
        refinement_steps=refinement_steps,
        sds_steps=sds_steps,
        sds_unet=sds_unet,
        sds_lr=sds_lr,
        t_min=t_min,
        t_max=t_max,
        use_condition_tpc=use_condition_tpc,
        condition_tpc_weight=condition_tpc_weight,
        condition_image=condition_image,
        fixed_slices=fixed_slices,
    )

    latent_index = voxel_to_latent_index(slice_index)
    condition_error = condition_error_from_volume(volume_z, condition_z, axis, slice_index)
    return {
        "volume_z": volume_z,
        "volume": volume,
        "condition_z": condition_z,
        "latent_index": latent_index,
        "condition_error": condition_error,
        "sds_history": sds_history,
    }


predict_conditioned_volume = predict


@torch.no_grad()
def predict_many(
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    conditions: list[dict[str, torch.Tensor | int]],
    condition_is_latent: bool = False,
    volume_shape: tuple[int, int, int, int] = (4, 16, 16, 16),
    refinement_steps: int = 0,
    sds_steps: int = 0,
    sds_unet: torch.nn.Module | None = None,
    sds_lr: float = 1e-2,
    t_min: int = 50,
    t_max: int = 950,
    use_condition_tpc: bool = False,
    condition_tpc_weight: float = 0.0,
    lock_condition_slice: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    device = torch.device(device)
    vae.eval()
    unet.eval()
    encoded = encode_condition_items(
        vae=vae,
        conditions=conditions,
        condition_is_latent=condition_is_latent,
        lock_condition_slice=lock_condition_slice,
        device=device,
    )

    volume_z = sample_conditioned_latent_volume_multi(
        unet=unet,
        ddpm=ddpm,
        condition_slices=encoded.condition_slices,
        volume_shape=volume_shape,
        device=device,
    )
    volume = multi_axis_decode(vae, volume_z)
    volume, sds_history = _refine_decoded_volume(
        volume=volume,
        vae=vae,
        unet=unet,
        ddpm=ddpm,
        refinement_steps=refinement_steps,
        sds_steps=sds_steps,
        sds_unet=sds_unet,
        sds_lr=sds_lr,
        t_min=t_min,
        t_max=t_max,
        use_condition_tpc=use_condition_tpc,
        condition_tpc_weight=condition_tpc_weight,
        condition_image=encoded.first_condition_image,
        fixed_slices=encoded.fixed_slices,
    )

    condition_errors = []
    for item in encoded.condition_slices:
        axis = int(item["axis"])
        slice_index = int(item["slice_index"])
        condition_errors.append(condition_error_from_volume(volume_z, item["condition_z"], axis, slice_index))

    return {
        "volume_z": volume_z,
        "volume": volume,
        "condition_slices": encoded.condition_slices,
        "condition_errors": condition_errors,
        "sds_history": sds_history,
    }


def predict_with_config(
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    config: PredictConfig,
) -> dict[str, object]:
    return predict_many(
        vae=vae,
        unet=unet,
        ddpm=ddpm,
        conditions=condition_specs_to_dicts(config.conditions),
        condition_is_latent=config.condition_is_latent,
        volume_shape=config.volume_shape,
        refinement_steps=config.refinement_steps,
        sds_steps=config.sds_steps,
        sds_unet=config.sds_unet,
        sds_lr=config.sds_lr,
        t_min=config.t_min,
        t_max=config.t_max,
        use_condition_tpc=config.use_condition_tpc,
        condition_tpc_weight=config.condition_tpc_weight,
        lock_condition_slice=config.lock_condition_slice,
        device=config.device,
    )

def predict_scale_up(
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    conditions: list[ConditionSpec],
    output_size: int | None = None,
    latent_ch: int = 4,
    downsample: int = 4,
    refinement_steps: int = 0,
    sds_steps: int = 0,
    sds_unet: torch.nn.Module | None = None,
    sds_lr: float = 1e-2,
    t_min: int = 50,
    t_max: int = 950,
    use_condition_tpc: bool = False,
    condition_tpc_weight: float = 0.0,
    lock_condition_slice: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    volume_shape = scale_up_volume_shape(
        conditions=conditions,
        output_size=output_size,
        latent_ch=latent_ch,
        downsample=downsample,
    )
    return predict_many(
        vae=vae,
        unet=unet,
        ddpm=ddpm,
        conditions=condition_specs_to_dicts(conditions),
        condition_is_latent=False,
        volume_shape=volume_shape,
        refinement_steps=refinement_steps,
        sds_steps=sds_steps,
        sds_unet=sds_unet,
        sds_lr=sds_lr,
        t_min=t_min,
        t_max=t_max,
        use_condition_tpc=use_condition_tpc,
        condition_tpc_weight=condition_tpc_weight,
        lock_condition_slice=lock_condition_slice,
        device=device,
    )


def predict_scale_up_with_config(
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    ddpm,
    config: ScaleUpConfig,
) -> dict[str, object]:
    return predict_scale_up(
        vae=vae,
        unet=unet,
        ddpm=ddpm,
        conditions=config.conditions,
        output_size=config.output_size,
        latent_ch=config.latent_ch,
        downsample=config.downsample,
        refinement_steps=config.refinement_steps,
        sds_steps=config.sds_steps,
        sds_unet=config.sds_unet,
        sds_lr=config.sds_lr,
        t_min=config.t_min,
        t_max=config.t_max,
        use_condition_tpc=config.use_condition_tpc,
        condition_tpc_weight=config.condition_tpc_weight,
        lock_condition_slice=config.lock_condition_slice,
        device=config.device,
    )
