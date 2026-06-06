"""
Smoke-test MicroLAD generation with the local microlad-anode weights.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe test.py

The defaults are intentionally light enough for a quick CPU check. Increase
--ddpm_steps, --sds_steps, --refinement_K, and the loss weights for a heavier
generation run.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "reference"
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

from generate import generate_single_volume  # noqa: E402
from losses import setup_tpc_bins  # noqa: E402
from models import CustomVAE, DDPM, TimeUNet, TorchFEMMesh  # noqa: E402
from utils import compute_volume_tpc, plot_tpc_comparison, visualize_3d_microstructure  # noqa: E402


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def parse_float_targets(text: str) -> dict[float, float]:
    targets: dict[float, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        targets[float(key)] = float(value)
    return targets


def parse_int_targets(text: str | None) -> dict[int, float] | None:
    if not text:
        return None
    return {int(k): float(v) for k, v in (item.split(":", 1) for item in text.split(","))}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def load_models(args: argparse.Namespace, device: torch.device) -> tuple[CustomVAE, TimeUNet]:
    vae = CustomVAE(latent_ch=args.latent_ch).to(device)
    vae_checkpoint = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(vae_checkpoint["vae"] if "vae" in vae_checkpoint else vae_checkpoint)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    unet = TimeUNet(args.latent_ch).to(device)
    unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    unet.eval()
    for param in unet.parameters():
        param.requires_grad_(False)

    return vae, unet


def load_tpc_targets(path: Path) -> tuple[list[int], dict[int, np.ndarray]]:
    train_data = np.load(path)
    phases = sorted(int(key[1]) for key in train_data.files if key.endswith("_mean"))
    targets = {phase: train_data[f"S{phase}{phase}_mean"] for phase in phases}
    return phases, targets


def save_slice_preview(volume: np.ndarray, save_path: Path) -> None:
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2
    slices = [
        ("Z mid", volume[z_mid, :, :]),
        ("Y mid", volume[:, y_mid, :]),
        ("X mid", volume[:, :, x_mid]),
    ]

    cmap = ListedColormap(["#333333", "#9a9a9a", "#f2f2f2"])
    norm = BoundaryNorm([-0.01, 0.25, 0.75, 1.01], cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for ax, (title, image) in zip(axes, slices, strict=True):
        ax.imshow(image, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one or more 3D MicroLAD test volumes.")

    parser.add_argument("--vae_ckpt", type=Path, default=ROOT / "microlad-anode" / "vae_anode.pth")
    parser.add_argument("--unet_ckpt", type=Path, default=ROOT / "microlad-anode" / "unet_anode.pth")
    parser.add_argument(
        "--training_tpc",
        type=Path,
        default=ROOT / "microlad-anode" / "autocorr_periodic_mean_std.npz",
    )
    parser.add_argument("--save_dir", type=Path, default=ROOT / "output" / "test_microstructure")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or a CUDA device such as 'cuda:0'.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_volumes", type=int, default=1)

    parser.add_argument("--ddpm_steps", type=int, default=25)
    parser.add_argument("--sds_steps", type=int, default=0)
    parser.add_argument("--sds_lr", type=float, default=0.001)
    parser.add_argument("--t_min", type=int, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--refinement_K", type=int, default=0)

    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--latent_ch", type=int, default=4)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--W", type=int, default=16)

    parser.add_argument("--vf_targets", default="0:0.35,0.5:0.28,1:0.37")
    parser.add_argument("--vf_weight", type=float, default=0.0)
    parser.add_argument("--tpc_weight", type=float, default=0.0)
    parser.add_argument("--rd_weight", type=float, default=0.0)
    parser.add_argument("--sa_weight", type=float, default=0.0)
    parser.add_argument("--rd_targets", default=None)
    parser.add_argument("--sa_targets", default=None)

    parser.add_argument("--skip_visualization", action="store_true")
    parser.add_argument("--plot_tpc", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.vae_ckpt = resolve_path(args.vae_ckpt)
    args.unet_ckpt = resolve_path(args.unet_ckpt)
    args.training_tpc = resolve_path(args.training_tpc)
    args.save_dir = resolve_path(args.save_dir)

    if args.num_samples != 16 or args.H != 16 or args.W != 16:
        raise ValueError("The reference 64x64x64 decoder path expects num_samples=H=W=16.")
    if args.ddpm_steps < 1:
        raise ValueError("--ddpm_steps must be at least 1.")

    args.t_min = min(200, max(0, args.ddpm_steps // 5)) if args.t_min is None else args.t_min
    args.t_max = min(500, args.ddpm_steps) if args.t_max is None else args.t_max
    if args.sds_steps > 0 and not (0 <= args.t_min < args.t_max <= args.ddpm_steps):
        raise ValueError("--t_min and --t_max must satisfy 0 <= t_min < t_max <= ddpm_steps.")

    vf_targets = parse_float_targets(args.vf_targets)
    args.vf0 = vf_targets.get(0.0, 0.35)
    args.vf05 = vf_targets.get(0.5, 0.28)
    args.vf1 = vf_targets.get(1.0, 0.37)
    args.w_m1 = args.vf_weight
    args.w_m2 = args.vf_weight
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    seed_everything(args.seed)
    device = choose_device(args.device)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    for required in (args.vae_ckpt, args.unet_ckpt, args.training_tpc):
        if not required.exists():
            raise FileNotFoundError(required)

    print(f"Device: {device}")
    print(f"VAE: {args.vae_ckpt}")
    print(f"UNet: {args.unet_ckpt}")
    print(f"TPC stats: {args.training_tpc}")
    print(f"DDPM steps: {args.ddpm_steps}, SDS steps: {args.sds_steps}, refinement_K: {args.refinement_K}")

    vae, unet = load_models(args, device)
    ddpm = DDPM(timesteps=args.ddpm_steps, beta_start=1e-4, beta_end=2e-2, device=device)
    phases, tpc_targets = load_tpc_targets(args.training_tpc)

    rd_targets = parse_int_targets(args.rd_targets)
    sa_targets = parse_int_targets(args.sa_targets)
    fem_solver = None
    if args.rd_weight > 0:
        fem_solver = TorchFEMMesh(M=64, N=64, low_cond=0.001, device=device).to(device)

    bin_mat = None
    bin_counts = None
    if args.sds_steps > 0 and args.tpc_weight > 0:
        bin_mat, bin_counts = setup_tpc_bins(64, 64, device)

    for volume_index in range(args.n_volumes):
        print(f"\nGenerating volume {volume_index + 1}/{args.n_volumes}")
        volume = generate_single_volume(
            vae,
            unet,
            ddpm,
            fem_solver,
            bin_mat,
            bin_counts,
            tpc_targets,
            sa_targets,
            rd_targets,
            phases,
            args,
            device,
        )

        out_dir = args.save_dir / f"volume_{volume_index:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        volume_path = out_dir / "volume.tiff"
        preview_path = out_dir / "middle_slices.png"
        tifffile.imwrite(volume_path, volume.astype(np.float32))
        save_slice_preview(volume, preview_path)
        print(f"Saved volume: {volume_path}")
        print(f"Saved slice preview: {preview_path}")

        if not args.skip_visualization:
            visualization_path = out_dir / "3d_visualization_final.png"
            visualize_3d_microstructure(volume, visualization_path)
            print(f"Saved 3D visualization: {visualization_path}")

        if args.plot_tpc:
            volume_labels = (volume * 2).astype(np.int32)
            tpc_results = compute_volume_tpc(volume_labels, phases)
            for axis_name in ("x", "y", "z"):
                axis_tpcs = {phase: tpc_results[(axis_name, phase)] for phase in phases}
                plot_tpc_comparison(
                    axis_tpcs,
                    tpc_targets,
                    phases,
                    out_dir / f"tpc_compare_{axis_name}.png",
                    axis_name,
                )
            print("Saved TPC comparison plots")

    print(f"\nDone. Results are under: {args.save_dir}")


if __name__ == "__main__":
    main()
