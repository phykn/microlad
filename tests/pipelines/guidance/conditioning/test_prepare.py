import unittest

import numpy as np
import torch

from src.app.api import AnchorSlice
from src.pipelines.guidance.conditioning.images import prepare_anchor_image
from src.pipelines.guidance.conditioning.prepare import (
    build_anchor_constraint_volume,
    build_anchor_targets,
)


class ShiftDecodeVAE(torch.nn.Module):
    image_size = 2
    latent_size = 2
    latent_ch = 1
    num_phases = 2

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image.clone(), torch.zeros_like(image)

    def decode_probs(self, latent: torch.Tensor) -> torch.Tensor:
        labels = 1 - latent[:, 0].long()
        return torch.nn.functional.one_hot(labels, num_classes=2).movedim(-1, 1).float()


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
        expected = 1.0 - raw
        self.assertTrue(torch.equal(targets[(0, 1)][0, 0], expected))

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

    def test_constraint_volume_rejects_conflicting_cross_axis_labels(self):
        anchors = [
            AnchorSlice(
                image=np.zeros((2, 2), dtype=np.uint8),
                axis=0,
                index=1,
            ),
            AnchorSlice(
                image=np.array([[0, 0], [1, 1]], dtype=np.uint8),
                axis=1,
                index=0,
            ),
        ]

        with self.assertRaisesRegex(ValueError, "Conflicting anchor intersection"):
            build_anchor_constraint_volume(
                BiasedCategoricalVAE(),
                anchors,
                volume_shape=torch.Size([2, 2, 2]),
                num_phases=2,
                segment=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
