from pathlib import Path

import torch

from ..diffusion import DDPMProcess
from ..misc import load_config
from ..model import MPDDUNet
from ..model.factory import build_mpdd_model
from .predictor import MPDDPredictor


def load_predictor(
    run_dir: str | Path,
    device: str | torch.device | None = None,
) -> MPDDPredictor:
    device = torch.device(
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )
    cfg = _load_config(run_dir)
    _require_keys(
        cfg,
        "model config",
        "timesteps",
        "beta_start",
        "beta_end",
    )
    model = build_mpdd_model(cfg).to(device)
    model = _load_model(
        model,
        _find_checkpoint(run_dir),
        device,
    )
    ddpm = DDPMProcess(
        timesteps=cfg["timesteps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        device=device,
    )
    return MPDDPredictor(
        model=model,
        ddpm=ddpm,
        image_size=model.image_size,
        num_phases=model.num_phases,
        device=device,
    )


def _load_config(run_dir: str | Path) -> dict:
    path = _require_file(Path(run_dir) / "model.yaml", "model config")
    return load_config(path, label="model config")


def _find_checkpoint(run_dir: str | Path) -> Path:
    return Path(run_dir) / "weight" / "mpdd" / "last" / "model.pt"


def _require_file(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is required: {path}")
    return path


def _require_keys(config: dict, label: str, *names: str) -> None:
    missing = [name for name in names if name not in config]
    if missing:
        raise ValueError(f"{label} is missing required value: {', '.join(missing)}")


def _load_model(
    model: MPDDUNet,
    path: str | Path,
    device: torch.device,
) -> MPDDUNet:
    path = _require_file(path, "MPDD checkpoint")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state = ckpt
        if isinstance(ckpt, dict):
            state = next(
                (ckpt[key] for key in ("model", "mpdd", "unet") if key in ckpt),
                ckpt,
            )
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(
            f"MPDD checkpoint could not be loaded for model: {path}"
        ) from exc

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
