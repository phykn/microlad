# Microlad

Microlad reconstructs categorical 3D microstructures from 2D micrographs with
one image-space diffusion model. During sampling, the model alternates over the
`xy`, `xz`, and `yz` planes of a shared 3D noise volume.

The maintained pipeline supports axis-conditioned training, soft slice anchors,
phase-fraction conditioning, DDIM sampling, harmonization, and overlapping-tile
scale-up. It does not require 3D training volumes or a VAE.

## Quick start

```powershell
python -m pip install -r requirements.txt
python gen_data.py
python run_train.py
```

`gen_data.py` reads `config/simul.yaml` and writes a synthetic volume, its three
orthogonal slice sets, and `data/generated/manifest.json`. It refuses to
overwrite non-empty output directories.

`run_train.py` reads `config/model.yaml`. Checkpoints are written under
`run/<timestamp>/weight/mpdd/`.

## Dataset contract

Axis-conditioned training reads one manifest:

```json
{
  "schema_version": 1,
  "volume_axes": "ZYX",
  "axis_sources": {
    "xy": "train/xy",
    "xz": "train/xz",
    "yz": "train/yz"
  }
}
```

Source directories are POSIX paths relative to the manifest. Plane conditions
are fixed:

- `xy = 0`, normal to `z`
- `xz = 1`, normal to `y`
- `yz = 2`, normal to `x`

Images must contain integer phase labels from `0` to `K - 1`. All three sources
must use the same phase meanings and physical resolution. Training samples the
three axes evenly, regardless of the number of images in each directory.

For an isotropic control, map the same normalized image distribution to all
three sources. For the synthetic anisotropic fixture, set
`geometry.shape: aligned_ellipsoid` in `config/simul.yaml`.

## Prediction

Set `model.run_dir` in `config/predict.yaml` to a trained run, then use the
maintained API:

```python
from src.predict import load_predict_config, load_predictor

config = load_predict_config("config/predict.yaml")
predictor = load_predictor(config.run_dir)
options = config.make_options(predictor)
volume, stats = predictor.predict(options)
```

`volume` is a `uint8` tensor with shape `[D, H, W] = [z, y, x]`.

## Core behavior

- **Axis conditions:** one shared U-Net receives the condition for the active
  plane. Phase-fraction classifier-free guidance does not drop this condition.
- **Soft anchors:** an integrated anchor encoder conditions predicted noise.
  Anchor pixels are not copied into the evolving or final volume.
- **Scale-up:** large planes are denoised as overlapping training-size tiles and
  merged with smooth weights.
- **Sampling:** DDPM and DDIM are supported. Harmonization repeats plane-wise
  refinement at a diffusion step.

Anchors require a newly trained anchor-enabled checkpoint. Older checkpoints
remain usable for ordinary generation but reject anchor inputs.

Runnable examples are in `notebooks/`:

- `00_dataset.ipynb`: manifest, dataset, and balanced axis batches
- `01_model.ipynb`: model and axis-condition checks
- `02_predict.ipynb`: anchored 3D prediction
- `03_scale_up.ipynb`: overlapping-tile generation

## Examples

Soft anchor continuity:

![Anchor input, generated anchor plane, and neighboring slices](docs/assets/anchor-neighborhood.png)

Orthogonal slices and phase surfaces from one generated volume:

![Orthogonal slices and 3D phase surfaces](docs/assets/generated-volume.png)

Overlapping-tile scale-up:

![Scaled volume slices and enlarged tile-overlap junctions](docs/assets/scale-up.png)

## Limits

- Matching 2D slice statistics does not identify a unique or necessarily
  physical 3D structure.
- Realistic slices do not guarantee correct connectivity, topology, or material
  properties.
- Phase fractions and anchors are learned conditions, not exact constraints.
- Tiled scale-up preserves local behavior more directly than long-range
  structure.
- Results should be checked with directional 2D statistics and relevant 3D
  connectivity or property metrics.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## References

- [Micro3Diff](https://doi.org/10.1038/s41524-024-01280-z)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
