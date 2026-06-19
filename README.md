# Conditioned MicroLad

MicroLad-style 2D-to-3D microstructure generation.

This repo trains on 2D grayscale image patches, then generates a 3D candidate. During inference, one or more observed 2D slices can be fixed inside the generated volume.

## Install

```sh
python -m pip install -r requirements.txt
```

## Config

- VAE config: `config/train_vae.yaml`
- UNet config: `config/train_unet.yaml`
- Put training images in `data.data_dir`.
- Train VAE first, then train UNet.
- `checkpoints.vae_ckpt` is required for UNet training.
- `checkpoints.unet_ckpt` is only for resuming a UNet.

## Train

```sh
python run_train_vae.py
python run_train_unet.py
```

DDP:

```sh
torchrun --nproc_per_node 4 run_train_vae.py
torchrun --nproc_per_node 4 run_train_unet.py
```

## Predict

```python
from src.inference import MicroLadPredictor
from src.models import DDPM

predictor = MicroLadPredictor(
    vae=vae,
    unet=unet,
    ddpm=DDPM(timesteps=1000, device=device),
    device=device,
)

volume = predictor.predict()["volume"]
```

With an observed slice:

```python
volume = predictor.predict({
    "size": 64,
    "images": [{"image": condition_image, "axis": 0, "index": 12}],
})["volume"]
```

`condition_image` can be a numpy array or torch tensor. It is converted to grayscale and normalized before entering the model.

## Data Shape

- 2D images: `H x W`
- 3D TIFF stack: `D x H x W`
- Dataset output: `[1, H, W]`, float, `0..1`
- Generated volume: `[D, 1, H, W]`

## Reference

- Original GitHub: [KangHyunL/microlad](https://github.com/KangHyunL/microlad)
- Paper: [MicroLad](https://arxiv.org/abs/2508.20138)
