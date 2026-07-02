import unittest

import numpy as np
import torch

from src.predict import AnchorSlice
from src.predict.sds.common import prepare_anchor_targets


class PredictSDSCommonTest(unittest.TestCase):
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
                anchors,
                volume_shape=torch.Size([2, 2, 2]),
                num_phases=4,
                segment=False,
                device=torch.device("cpu"),
                dtype=torch.long,
            )


if __name__ == "__main__":
    unittest.main()
