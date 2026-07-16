import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gen_data as script


class GenerateDataTest(unittest.TestCase):
    def test_parse_args_preserves_generator_and_export_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "simul.yaml"
            config.write_text(
                "output:\n"
                "  data_dir: generated\n"
                "  count: 2\n"
                "geometry:\n"
                "  mode: dry\n"
                "  size: 32\n"
                "  shape: aligned_ellipsoid\n"
                "export:\n"
                "  planes: [xy, xz, yz]\n",
                encoding="utf-8",
            )
            with patch.object(script, "DEFAULT_CONFIG", str(config)):
                args = script.parse_args([])

        self.assertEqual(
            args.output,
            {"data_dir": (Path(tmp) / "generated").resolve(), "count": 2},
        )
        self.assertEqual(args.geometry["mode"], "dry")
        self.assertEqual(args.geometry["shape"], "aligned_ellipsoid")
        self.assertNotIn("seed", args.geometry)
        self.assertEqual(args.export["planes"], ["xy", "xz", "yz"])

    def test_main_forwards_nested_sections_to_simulation_facade(self):
        data_dir = Path("generated").resolve()
        args = type(
            "Args",
            (),
            {
                "output": {"data_dir": data_dir, "count": 3},
                "geometry": {"mode": "dry", "size": 32},
                "export": {"planes": ["xy", "xz", "yz"]},
            },
        )()
        volumes = [data_dir / "gt" / "volume_000.tif"]
        planes = {
            "xy": [data_dir / "train" / "xy" / "slice_z_000.png"],
            "xz": [data_dir / "train" / "xz" / "slice_y_000.png"],
            "yz": [data_dir / "train" / "yz" / "slice_x_000.png"],
        }

        with (
            patch.object(script, "parse_args", return_value=args),
            patch.object(
                script,
                "save_simulation",
                return_value=(volumes, planes),
            ) as save,
            patch("builtins.print"),
        ):
            script.main()

        save.assert_called_once_with(
            data_dir,
            count=3,
            geometry=args.geometry,
            export=args.export,
        )


if __name__ == "__main__":
    unittest.main()
