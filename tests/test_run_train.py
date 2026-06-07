import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import run_train_vae
import run_train_unet
from src import build
from src.dataset import PatchDataset
from src.models import CustomVAE, DDPM, TimeUNet

import numpy as np
from PIL import Image


class RunTrainTest(unittest.TestCase):
    def _write_image(self, image_dir: str) -> None:
        os.makedirs(image_dir, exist_ok=True)
        Image.fromarray(np.full((64, 64), 128, dtype=np.uint8), mode="L").save(os.path.join(image_dir, "sem.png"))

    def test_root_run_train_builds_trainer_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = os.path.join(tmp, "images")
            self._write_image(image_dir)

            dataset = PatchDataset(
                root_dir=image_dir,
                patch_size=64,
                seed=0,
            )
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            vae = CustomVAE(latent_ch=4).eval()
            unet = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
            ddpm = DDPM(timesteps=10)
            optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)

            trainer = build.build_trainer(
                unet=unet,
                vae=vae,
                ddpm=ddpm,
                loader=loader,
                optimizer=optimizer,
                scheduler=None,
                save_dir=tmp,
                max_grad_norm=1.0,
                accum_steps=1,
                rank=0,
            )

            losses = trainer.step()

            self.assertIn("loss", losses)

    def test_parse_args_supports_birefnet_style_steps_flags(self):
        argv = [
            "run_train_unet.py",
            "--data-dir",
            "data",
            "--vae-ckpt",
            "microlad-anode/vae_anode.pth",
            "--steps",
            "1",
        ]

        with patch("sys.argv", argv):
            args = run_train_unet.parse_args()

        self.assertEqual(args.steps, 1)
        self.assertEqual(args.data_dir, "data")
        self.assertEqual(args.output_dir, "output/unet")

    def test_build_unet_can_load_state_dict_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "unet.pth")
            source = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
            torch.save(source.state_dict(), ckpt_path)
            args = run_train_unet.parse_args_from_list(
                [
                    "--data-dir",
                    "data",
                    "--vae-ckpt",
                    "microlad-anode/vae_anode.pth",
                    "--unet-ckpt",
                    ckpt_path,
                    "--base-ch",
                    "16",
                    "--time-dim",
                    "16",
                ]
            )

            unet = build.build_unet(args, torch.device("cpu"))

            self.assertTrue(torch.allclose(unet.enc2.conv1.weight, source.enc2.conv1.weight))

    def test_build_unet_can_load_trainer_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "last.pth")
            source = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
            torch.save({"model": source.state_dict()}, ckpt_path)
            args = run_train_unet.parse_args_from_list(
                [
                    "--data-dir",
                    "data",
                    "--vae-ckpt",
                    "microlad-anode/vae_anode.pth",
                    "--unet-ckpt",
                    ckpt_path,
                    "--base-ch",
                    "16",
                    "--time-dim",
                    "16",
                ]
            )

            unet = build.build_unet(args, torch.device("cpu"))

            self.assertTrue(torch.allclose(unet.out.weight, source.out.weight))

    def test_run_train_vae_parse_args(self):
        args = run_train_vae.parse_args_from_list(
            [
                "--data-dir",
                "data",
                "--output-dir",
                "output/vae",
                "--steps",
                "1",
            ]
        )

        self.assertEqual(args.steps, 1)
        self.assertEqual(args.data_dir, "data")

    def test_run_train_vae_detects_distributed_environment(self):
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"}):
            self.assertTrue(build.is_distributed())

    def test_run_train_unet_loads_yaml_config_with_cli_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "train_unet.yaml")
            with open(config, "w", encoding="utf-8") as f:
                f.write("data_dir: data\nvae_ckpt: microlad-anode/vae_anode.pth\nsteps: 10\n")

            args = run_train_unet.parse_args_from_list(["--config", config, "--steps", "2"])

            self.assertEqual(args.data_dir, "data")
            self.assertEqual(args.vae_ckpt, "microlad-anode/vae_anode.pth")
            self.assertEqual(args.steps, 2)


if __name__ == "__main__":
    unittest.main()
