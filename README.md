# Conditioned MicroLad

Conditioned MicroLad is an experimental extension of the original MicroLad pipeline for generating 3D microstructure candidates from 2D SEM image patches. It adds observation conditioning so one or more real 2D slices can be fixed at specified internal positions during generation.

The core flow is:

1. Put source SEM images in the image directory used by the training config.
2. The dataset reads the original images and randomly crops 64x64 patches in memory.
3. The VAE encodes each 2D patch into a latent representation.
4. The UNet/DDPM learns to predict noise in the 2D latent space.
5. Inference builds a random 3D latent candidate and can lock observed slices at specified axes and indices.

## Installation

```sh
python -m pip install -r requirements.txt
```

## Training

Train the VAE:

```sh
python run_train_vae.py --config src/config/train_vae.yaml
```

Train the UNet:

```sh
python run_train_unet.py --config src/config/train_unet.yaml
```

The UNet learns noise prediction on 2D latents produced by the VAE. Observation images and their positions are not part of the training loop; they are applied during inference.

For distributed training, use `torchrun` on a GPU server:

```sh
torchrun --nproc_per_node 4 run_train_vae.py --config src/config/train_vae.yaml
torchrun --nproc_per_node 4 run_train_unet.py --config src/config/train_unet.yaml
```

The same trained VAE and UNet are used for larger generated candidates. Pass `condition={"size": 512}` to `MicroLadPredictor.predict()` to generate a 512x512x512 candidate without observation slices. Add `images` to lock one or more observed slices. Large condition images are internally split into 64x64 latent tiles.

## Prediction

Use `MicroLadPredictor.predict()`. It returns a dictionary with `volume` and `sds_history`. The example below assumes that trained `vae`, `unet`, and `device` objects are already available.

```python
import numpy as np
from PIL import Image

from src.inference import MicroLadPredictor
from src.models import DDPM

image = Image.open("data/sample_01.png").convert("L").resize((64, 64))
condition_image = np.asarray(image, dtype=np.float32)

predictor = MicroLadPredictor(
    vae=vae,
    unet=unet,
    ddpm=DDPM(timesteps=1000, device=device),
    device=device,
)
result = predictor.predict({
    "size": 64,
    "images": [{"image": condition_image, "axis": 0, "index": 12}],
})

volume = result["volume"]
```

Generate without observation conditions:

```python
volume = predictor.predict()["volume"]
```

Set the output size with `size`. For example, `{"size": 512}` creates a 512x512x512 candidate. When `images` is provided, each item fixes one slice at the given `axis` and `index`. Condition images are resized to the requested `size` when needed.

Use `GenerationOptions(sds_steps=...)` only when SDS refinement is needed. In that case, `condition_weight` makes the result follow the fixed condition slices more directly, and `stats_weight` automatically computes VF/TPC/SA statistics from the condition images and matches them during refinement.

## Notebooks

Useful notebooks:

- `notebooks/00_patch_dataset.ipynb`: checks random crops from source images
- `notebooks/02_ldm_forward.ipynb`: checks the 2D latent diffusion forward path
- `notebooks/07_generate_pipeline.ipynb`: checks generation with loaded weights
- `notebooks/09_scale_up_smoke.ipynb`: smoke test for large candidate generation
- `notebooks/10_reference_generate_smoke.ipynb`: smoke test with reference weights
- `notebooks/11_microstructure_viewer.ipynb`: slice viewer for generated TIFF volumes

Notebooks are kept short and focused on feature checks.

## Reference

- Original GitHub: [KangHyunL/microlad](https://github.com/KangHyunL/microlad)
- Paper: [MicroLad: 2D-to-3D Microstructure Reconstruction and Generation via Latent Diffusion and Score Distillation](https://arxiv.org/abs/2508.20138)
