import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.train.utils import save_checkpoint


class SaveCheckpointTest(unittest.TestCase):
    def test_last_checkpoint_is_not_written_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight_dir = root / "weight"
            last_weight_dir = weight_dir / "last"
            last_weight_dir.mkdir(parents=True)
            final_path = last_weight_dir / "model.pt"

            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            saved_paths = []

            def fake_save(checkpoint, path):
                path = Path(path)
                saved_paths.append(path)
                if path == final_path:
                    raise RuntimeError("open file failed with error code: 1224")
                path.write_bytes(b"checkpoint")

            with patch("src.train.utils.torch.save", side_effect=fake_save):
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=1,
                    save_every=2,
                    weight_dir=weight_dir,
                    last_weight_dir=last_weight_dir,
                    is_main_process=True,
                )

            self.assertTrue(final_path.is_file())
            self.assertNotIn(final_path, saved_paths)
