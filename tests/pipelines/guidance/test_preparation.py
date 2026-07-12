import unittest

import numpy as np
import torch

from src.app.api import AnchorSlice
from src.pipelines.guidance.conditioning.images import prepare_anchor_image
from src.pipelines.guidance.preparation import (
    build_anchor_constraint_volume,
    build_anchor_targets,
)


class ShiftDecodeVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.clone(), torch.zeros_like(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + 0.25


class BiasedCategoricalVAE(ShiftDecodeVAE):
    num_phases = 2

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        probabilities = torch.zeros(latent.shape[0], 2, 2, 2)
        probabilities[:, 0] = 1.0
        return probabilities


class PredictSDSCommonTest(unittest.TestCase):
    def test_build_anchor_targets_uses_vae_reconstruction(self):
        anchor = AnchorSlice(
            image=np.array([[0, 1], [1, 0]], dtype=np.uint8),
            axis=0,
            index=1,
        )

        targets = build_anchor_targets(
            ShiftDecodeVAE(),
            [anchor],
            volume_shape=torch.Size([2, 2, 2]),
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        raw = prepare_anchor_image(anchor.image, num_phases=2)[0, 0]
        expected = raw + 0.25
        self.assertTrue(torch.allclose(targets[(0, 1)][0, 0], expected))

    def test_build_anchor_targets_rejects_non_floating_dtype(self):
        anchors = [
            AnchorSlice(
                image=np.array([[0, 1], [2, 3]], dtype=np.uint8),
                axis=0,
                index=0,
            )
        ]

        with self.assertRaisesRegex(ValueError, "dtype"):
            build_anchor_targets(
                ShiftDecodeVAE(),
                anchors,
                volume_shape=torch.Size([2, 2, 2]),
                num_phases=4,
                segment=False,
                device=torch.device("cpu"),
                dtype=torch.long,
            )

    def test_categorical_constraint_volume_preserves_raw_anchor_labels(self):
        anchor = AnchorSlice(
            image=np.ones((2, 2), dtype=np.uint8),
            axis=0,
            index=1,
        )

        target, mask = build_anchor_constraint_volume(
            BiasedCategoricalVAE(),
            [anchor],
            volume_shape=torch.Size([2, 2, 2]),
            num_phases=2,
            segment=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertTrue(torch.all(target[1] == 1))
        self.assertTrue(torch.all(mask[1] == 1))
        self.assertTrue(torch.all(mask[0] == 0))


if __name__ == "__main__":
    unittest.main()
