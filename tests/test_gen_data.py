import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gen_data as script


class GenerateDataTest(unittest.TestCase):
    def test_parse_args_preserves_output_and_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "simul.yaml"
            path.write_text(
                "output:\n"
                "  data_dir: generated\n"
                "  count: 2\n"
                "  axes: [0, 1, 2]\n"
                "geometry:\n"
                "  size: 32\n"
                "  big_elongation: 2.0\n",
                encoding="utf-8",
            )
            with patch.object(script, "DEFAULT_CONFIG", str(path)):
                args = script.parse_args([])

        self.assertEqual(
            args.output,
            {
                "data_dir": (Path(tmp) / "generated").resolve(),
                "count": 2,
                "axes": [0, 1, 2],
            },
        )
        self.assertEqual(args.geometry["big_elongation"], 2.0)

    def test_main_forwards_nested_sections_to_simulation_facade(self):
        data_dir = Path("generated").resolve()
        args = type(
            "Args",
            (),
            {
                "output": {
                    "data_dir": data_dir,
                    "count": 3,
                    "axes": [0, 1, 2],
                },
                "geometry": {"size": 32},
            },
        )()
        vols = [data_dir / "gt" / "volume_000.tif"]
        slices = {
            0: [data_dir / "train" / "0" / "slice_0_000.png"],
            1: [data_dir / "train" / "1" / "slice_1_000.png"],
            2: [data_dir / "train" / "2" / "slice_2_000.png"],
        }

        with (
            patch.object(script, "parse_args", return_value=args),
            patch.object(
                script,
                "save_simulation",
                return_value=(vols, slices),
            ) as save,
            patch("builtins.print"),
        ):
            script.main()

        save.assert_called_once_with(
            data_dir,
            count=3,
            geometry=args.geometry,
            axes=args.output["axes"],
        )


if __name__ == "__main__":
    unittest.main()
