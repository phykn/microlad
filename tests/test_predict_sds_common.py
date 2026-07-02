import unittest

import numpy as np
import torch

from src.predict import AnchorSlice
from src.predict.anchor import prepare_anchor_image
from src.predict.sds.common import prepare_anchor_targets


class ShiftDecodeVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.clone(), torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + 0.25


class PredictSDSCommonTest(unittest.TestCase):
    def test_prepare_anchor_targets_uses_vae_reconstruction(self):
        anchor = AnchorSlice(
            image=np.array([[0, 1], [1, 0]], dtype=np.uint8),
            axis=0,
            index=1,
        )

        targets = prepare_anchor_targets(
            ShiftDecodeVAE(),
            [anchor],
            volume_shape=torch.Size([2, 2, 2]),
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        raw = prepare_anchor_image(anchor.image, num_phases=2)[0, 0]
        expected = (raw + 0.25).clamp(-1.0, 1.0)
        self.assertTrue(torch.allclose(targets[(0, 1)][0, 0], expected))

    def test_prepare_anchor_targets_rejects_non_floating_dtype(self):
        anchors = [
            AnchorSlice(
                image=np.array([[0, 1], [2, 3]], dtype=np.uint8),
                axis=0,
                index=0,
            )
        ]

        with self.assertRaisesRegex(ValueError, "dtype"):
            prepare_anchor_targets(
                ShiftDecodeVAE(),
                anchors,
                volume_shape=torch.Size([2, 2, 2]),
                num_phases=4,
                segment=False,
                device=torch.device("cpu"),
                dtype=torch.long,
            )


if __name__ == "__main__":
    unittest.main()
