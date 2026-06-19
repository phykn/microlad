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
from src.models import DDPM, PatchVAE, TimeUNet

import numpy as np
from PIL import Image


class RunTrainTest(unittest.TestCase):
    def _write_image(self, image_dir: str) -> None:
        os.makedirs(image_dir, exist_ok=True)
        Image.fromarray(np.full((64, 64), 128, dtype=np.uint8), mode="L").save(
            os.path.join(image_dir, "sem.png")
        )

    def _unet_args(self, **overrides):
        args = run_train_unet.parse_args_from_list([])
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_root_run_train_builds_trainer_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = os.path.join(tmp, "images")
            self._write_image(image_dir)

            dataset = PatchDataset(
                root_dir=image_dir,
                patch_size=64,
            )
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            vae = PatchVAE(latent_ch=4).eval()
            unet = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
            ddpm = DDPM(timesteps=10)
            optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)

            trainer = build.build_unet_trainer(
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

    def test_build_unet_can_load_state_dict_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "unet.pth")
            source = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
            torch.save(source.state_dict(), ckpt_path)
            args = self._unet_args(
                unet_ckpt=ckpt_path,
                base_ch=16,
                time_dim=16,
            )

            unet = build.load_unet(args, torch.device("cpu"))

            self.assertTrue(
                torch.allclose(unet.enc2.conv1.weight, source.enc2.conv1.weight)
            )

    def test_build_unet_can_load_trainer_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "last.pth")
            source = TimeUNet(latent_ch=4, base_ch=16, time_dim=16)
            torch.save({"model": source.state_dict()}, ckpt_path)
            args = self._unet_args(
                unet_ckpt=ckpt_path,
                base_ch=16,
                time_dim=16,
            )

            unet = build.load_unet(args, torch.device("cpu"))

            self.assertTrue(torch.allclose(unet.out.weight, source.out.weight))

    def test_build_vae_can_load_trainer_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "last.pth")
            source = PatchVAE(latent_ch=4)
            torch.save({"model": source.state_dict()}, ckpt_path)
            args = self._unet_args(vae_ckpt=ckpt_path)

            vae = build.load_frozen_vae(args, torch.device("cpu"))

            self.assertTrue(torch.allclose(vae.conv_in.weight, source.conv_in.weight))
            self.assertFalse(any(param.requires_grad for param in vae.parameters()))

    def test_run_train_vae_parse_args(self):
        args = run_train_vae.parse_args_from_list([])

        self.assertEqual(args.steps, 1000)
        self.assertEqual(args.data_dir, "data")
        self.assertEqual(args.output_dir, "output/vae")

    def test_run_train_unet_uses_default_yaml_config(self):
        args = run_train_unet.parse_args_from_list([])

        self.assertEqual(args.data_dir, "data")
        self.assertEqual(args.vae_ckpt, "microlad-anode/vae_anode.pth")
        self.assertIsNone(args.unet_ckpt)
        self.assertEqual(args.output_dir, "output/unet")

    def test_run_train_unet_rejects_config_argument(self):
        with self.assertRaises(SystemExit):
            run_train_unet.parse_args_from_list(["--config", "config/train_unet.yaml"])

    def test_run_train_unet_rejects_cli_overrides(self):
        with self.assertRaises(SystemExit):
            run_train_unet.parse_args_from_list(["--steps", "1"])

    def test_run_train_vae_rejects_all_arguments(self):
        args = run_train_vae.parse_args_from_list([])

        self.assertFalse(hasattr(args, "config"))
        with self.assertRaises(SystemExit):
            run_train_vae.parse_args_from_list(["--config", "config/train_vae.yaml"])
        with self.assertRaises(SystemExit):
            run_train_vae.parse_args_from_list(["--steps", "1"])

    def test_run_train_vae_detects_distributed_environment(self):
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"}):
            self.assertTrue(build.is_distributed())

    def test_setup_device_uses_local_rank_for_ddp_device(self):
        env = {"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "1"}
        with (
            patch.dict(os.environ, env),
            patch("src.build.torch.cuda.is_available", return_value=True),
            patch("src.build.dist.is_nccl_available", return_value=True),
            patch("src.build.torch.cuda.set_device") as set_device,
            patch("src.build.dist.init_process_group") as init_process_group,
        ):
            device, local_rank, distributed = build.setup_device()

        self.assertTrue(distributed)
        self.assertEqual(device, torch.device("cuda", 1))
        self.assertEqual(local_rank, 1)
        set_device.assert_called_once_with(1)
        init_process_group.assert_called_once_with(backend="nccl")

    def test_setup_device_supports_cpu_distributed_environment(self):
        env = {"RANK": "0", "WORLD_SIZE": "1"}
        with (
            patch.dict(os.environ, env, clear=False),
            patch("src.build.torch.cuda.is_available", return_value=False),
            patch("src.build.torch.cuda.set_device") as set_device,
            patch("src.build.dist.init_process_group") as init_process_group,
        ):
            device, local_rank, distributed = build.setup_device()

        self.assertTrue(distributed)
        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(local_rank, 0)
        set_device.assert_not_called()
        init_process_group.assert_called_once_with(backend="gloo")

    def test_wrap_distributed_omits_device_ids_for_cpu_model(self):
        model = torch.nn.Linear(1, 1)

        with patch("src.build.DistributedDataParallel") as ddp:
            build.wrap_distributed(model, local_rank=0, distributed=True)

        ddp.assert_called_once_with(model)

    def test_run_train_vae_cleans_up_distributed_on_error(self):
        with (
            patch("run_train_vae.parse_args") as parse_args,
            patch(
                "run_train_vae.setup_device",
                return_value=(torch.device("cpu"), 0, True),
            ),
            patch("run_train_vae.build_dataset", side_effect=RuntimeError("failed")),
            patch("run_train_vae.cleanup_distributed") as cleanup,
        ):
            parse_args.return_value = run_train_vae.parse_args_from_list([])

            with self.assertRaisesRegex(RuntimeError, "failed"):
                run_train_vae.main()

        cleanup.assert_called_once_with(True)

    def test_run_train_unet_cleans_up_distributed_on_error(self):
        with (
            patch("run_train_unet.parse_args") as parse_args,
            patch(
                "run_train_unet.setup_device",
                return_value=(torch.device("cpu"), 0, True),
            ),
            patch("run_train_unet.build_dataset", side_effect=RuntimeError("failed")),
            patch("run_train_unet.cleanup_distributed") as cleanup,
        ):
            parse_args.return_value = run_train_unet.parse_args_from_list([])

            with self.assertRaisesRegex(RuntimeError, "failed"):
                run_train_unet.main()

        cleanup.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
