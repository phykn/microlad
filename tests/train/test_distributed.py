import os
import unittest
from unittest.mock import patch

import torch

from src.train.distributed import (
    cleanup,
    setup,
    wrap,
)


class DistributedTest(unittest.TestCase):
    def test_setup_uses_plain_device_without_distributed_env(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "src.train.distributed.torch.cuda.is_available",
                return_value=False,
            ),
        ):
            device, local_rank, distributed = setup()

        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(local_rank, 0)
        self.assertFalse(distributed)

    def test_setup_initializes_cpu_distributed(self):
        environment = {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch(
                "src.train.distributed.torch.cuda.is_available",
                return_value=False,
            ),
            patch("src.train.distributed.dist.init_process_group") as initialize,
        ):
            device, local_rank, distributed = setup()

        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(local_rank, 0)
        self.assertTrue(distributed)
        initialize.assert_called_once_with(backend="gloo")

    def test_wrap_and_cleanup(self):
        model = torch.nn.Linear(2, 2)
        with patch(
            "src.train.distributed.DistributedDataParallel"
        ) as distributed_model:
            wrapped = wrap(model, local_rank=0, enabled=True)
        distributed_model.assert_called_once_with(model)
        self.assertIs(wrapped, distributed_model.return_value)

        with (
            patch("src.train.distributed.dist.is_available", return_value=True),
            patch("src.train.distributed.dist.is_initialized", return_value=True),
            patch("src.train.distributed.dist.destroy_process_group") as destroy,
        ):
            cleanup(True)
        destroy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
