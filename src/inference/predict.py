from dataclasses import dataclass, field

import torch

from loss import build_grayscale_tpc_target
from .slice_conditioned import sample_conditioned_latent_volume, sample_conditioned_latent_volume_multi, voxel_to_latent_index
from .volume import multi_axis_decode, sds_refine_volume, three_axis_refinement


@dataclass
class ConditionSpec:
    condition: torch.Tensor
    axis: int
    slice_index: int


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

    condition_image = None
    if condition_is_latent:
        condition_z = condition.to(device)
        if condition_z.ndim == 4:
            condition_z = condition_z.squeeze(0)
    else:
        vae.eval()
        condition = condition.to(device)
        if condition.ndim == 3:
            condition = condition.unsqueeze(0)
        condition_image = condition
        condition_z, _ = vae.encode(condition * 2 - 1)
        condition_z = condition_z.squeeze(0)

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
    if refinement_steps > 0:
        volume = three_axis_refinement(volume, vae, refinement_steps=refinement_steps)
    sds_history = []
    if sds_steps > 0:
        denoise_unet = sds_unet if sds_unet is not None else unet
        grayscale_tpc_target = None
        grayscale_tpc_bin_mat = None
        grayscale_tpc_bin_counts = None
        if use_condition_tpc:
            if condition_image is None:
                raise ValueError("use_condition_tpc requires image-space condition input.")
            grayscale_tpc_target, grayscale_tpc_bin_mat, grayscale_tpc_bin_counts = build_grayscale_tpc_target(
                condition_image
            )
        fixed_slices = None
        if lock_condition_slice and condition_image is not None:
            fixed_slices = [{"axis": axis, "index": slice_index, "image": condition_image.squeeze(0)}]
        volume, sds_history = sds_refine_volume(
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

    latent_index = voxel_to_latent_index(slice_index)
    if axis == 0:
        fixed = volume_z[:, latent_index, :, :]
    elif axis == 1:
        fixed = volume_z[:, :, latent_index, :]
    else:
        fixed = volume_z[:, :, :, latent_index]

    condition_error = float((fixed - condition_z).abs().max())
    return {
        "volume_z": volume_z,
        "volume": volume,
        "condition_z": condition_z,
        "latent_index": latent_index,
        "condition_error": condition_error,
        "sds_history": sds_history,
    }


predict_conditioned_volume = predict


def _condition_specs_to_dicts(conditions: list[ConditionSpec]) -> list[dict[str, torch.Tensor | int]]:
    return [
        {"condition": item.condition, "axis": item.axis, "slice_index": item.slice_index}
        for item in conditions
    ]


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
    if not conditions:
        raise ValueError("conditions must not be empty.")

    device = torch.device(device)
    vae.eval()
    unet.eval()
    condition_slices = []
    fixed_slices = []
    first_condition_image = None

    for item in conditions:
        axis = int(item["axis"])
        slice_index = int(item["slice_index"])
        condition = item["condition"]
        if not isinstance(condition, torch.Tensor):
            raise ValueError("condition item must include tensor condition.")

        if condition_is_latent:
            condition_z = condition.to(device)
            if condition_z.ndim == 4:
                condition_z = condition_z.squeeze(0)
        else:
            condition_image = condition.to(device)
            if condition_image.ndim == 3:
                condition_image = condition_image.unsqueeze(0)
            if first_condition_image is None:
                first_condition_image = condition_image
            condition_z, _ = vae.encode(condition_image * 2 - 1)
            condition_z = condition_z.squeeze(0)
            if lock_condition_slice:
                fixed_slices.append({"axis": axis, "index": slice_index, "image": condition_image.squeeze(0)})

        condition_slices.append({"condition_z": condition_z, "axis": axis, "slice_index": slice_index})

    volume_z = sample_conditioned_latent_volume_multi(
        unet=unet,
        ddpm=ddpm,
        condition_slices=condition_slices,
        volume_shape=volume_shape,
        device=device,
    )
    volume = multi_axis_decode(vae, volume_z)
    if refinement_steps > 0:
        volume = three_axis_refinement(volume, vae, refinement_steps=refinement_steps)

    sds_history = []
    if sds_steps > 0:
        denoise_unet = sds_unet if sds_unet is not None else unet
        grayscale_tpc_target = None
        grayscale_tpc_bin_mat = None
        grayscale_tpc_bin_counts = None
        if use_condition_tpc:
            if first_condition_image is None:
                raise ValueError("use_condition_tpc requires image-space condition input.")
            grayscale_tpc_target, grayscale_tpc_bin_mat, grayscale_tpc_bin_counts = build_grayscale_tpc_target(
                first_condition_image
            )
        volume, sds_history = sds_refine_volume(
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
    elif fixed_slices:
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

    condition_errors = []
    for item in condition_slices:
        latent_index = voxel_to_latent_index(int(item["slice_index"]))
        axis = int(item["axis"])
        if axis == 0:
            fixed = volume_z[:, latent_index, :, :]
        elif axis == 1:
            fixed = volume_z[:, :, latent_index, :]
        else:
            fixed = volume_z[:, :, :, latent_index]
        condition_errors.append(float((fixed - item["condition_z"]).abs().max()))

    return {
        "volume_z": volume_z,
        "volume": volume,
        "condition_slices": condition_slices,
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
        conditions=_condition_specs_to_dicts(config.conditions),
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


def _infer_scale_up_size(conditions: list[ConditionSpec], output_size: int | None, downsample: int) -> int:
    if not conditions:
        raise ValueError("conditions must not be empty.")
    if downsample <= 0:
        raise ValueError("downsample must be positive.")

    first = conditions[0].condition
    if first.ndim == 4:
        _, _, h, w = first.shape
    elif first.ndim == 3:
        _, h, w = first.shape
    else:
        raise ValueError("condition must have shape [C, H, W] or [B, C, H, W].")
    if h != w:
        raise ValueError("scale-up condition crop must be square.")

    size = int(output_size) if output_size is not None else int(h)
    if size <= 0 or size % downsample != 0:
        raise ValueError("output_size must be positive and divisible by downsample.")
    if h != size or w != size:
        raise ValueError("condition crop size must match output_size.")

    for item in conditions:
        condition = item.condition
        if condition.ndim == 4:
            _, _, current_h, current_w = condition.shape
        elif condition.ndim == 3:
            _, current_h, current_w = condition.shape
        else:
            raise ValueError("condition must have shape [C, H, W] or [B, C, H, W].")
        if current_h != size or current_w != size:
            raise ValueError("all scale-up conditions must match output_size.")
        if item.slice_index < 0 or item.slice_index >= size:
            raise ValueError("condition slice_index must be inside output_size.")
    return size


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
    size = _infer_scale_up_size(conditions, output_size=output_size, downsample=downsample)
    latent_size = size // downsample
    return predict_many(
        vae=vae,
        unet=unet,
        ddpm=ddpm,
        conditions=_condition_specs_to_dicts(conditions),
        condition_is_latent=False,
        volume_shape=(latent_ch, latent_size, latent_size, latent_size),
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
