import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import PatchDataset
from models import CustomVAE
from training import Trainer
from training.loss import VAELoss


def parse_args_from_list(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    defaults = {}
    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="output/vae")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--latent-ch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--val-freq", type=int, default=500)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("--data-dir is required unless provided by --config")
    return args


def parse_args() -> argparse.Namespace:
    return parse_args_from_list()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PatchDataset(args.data_dir, patch_size=args.patch_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    vae = CustomVAE(latent_ch=args.latent_ch).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.min_lr,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=vae,
        train_loader=loader,
        valid_loader=None,
        criterion=VAELoss(kl_weight=args.kl_weight),
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=output_dir,
        max_grad_norm=args.max_grad_norm,
        accum_steps=args.accum_steps,
    )
    print(f"Training VAE steps={args.steps} save_dir={output_dir}")
    trainer.train(steps=args.steps, val_freq=args.val_freq, save_freq=args.save_freq)


if __name__ == "__main__":
    main()
