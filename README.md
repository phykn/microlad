# MicroLad

MicroLad는 2D SEM 이미지 조각으로 3D 미세구조 후보를 생성하는 실험용 구현체입니다. 이 저장소는 원본 MicroLad 흐름을 구현하면서, 실제 관측한 2D 조각 이미지를 특정 내부 위치에 조건으로 넣는 기능을 확장합니다.

핵심 흐름은 간단합니다.

1. 원본 SEM 이미지를 `data/images`에 넣는다.
2. 데이터셋이 원본 이미지를 읽고 메모리에서 64x64 crop을 랜덤으로 뽑는다.
3. VAE가 2D 조각 이미지를 latent로 압축한다.
4. UNet/DDPM이 3D latent 후보를 생성한다.
5. inference에서 조건 이미지와 위치를 넣어 3D 후보를 만들고, 필요하면 SDS와 multi-axis decode로 보정한다.

## 설치

Windows PowerShell 기준입니다.

```powershell
cd D:\code\microlad
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pillow pyyaml tensorboard matplotlib tifffile tqdm ipywidgets nbconvert nbclient jupyter
```

CUDA용 PyTorch가 필요하면 `torch torchvision` 부분은 본인 GPU/CUDA 버전에 맞는 PyTorch 공식 설치 명령으로 바꾸는 것이 좋습니다.

이 저장소는 아직 설치형 패키지로 묶지 않았기 때문에 실행 전에 `src`를 Python 경로에 넣습니다.

```powershell
$env:PYTHONPATH = "src"
```

## 데이터와 파일 위치

- `data/images`: 학습에 사용할 SEM 이미지 폴더
- `microlad-anode`: 기존 anode weight와 통계 파일 위치
- `output`: 학습 결과, weight, notebook 실행 결과 저장 위치
- `notebooks`: 기능별 최소 검증 노트북

`microlad-anode`, `output`, `reference`, `docs`는 git에 올리지 않는 로컬 작업 파일로 둡니다.

## 학습

VAE 학습:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe run_train_vae.py --config src/config/train_vae.yaml
```

조건부 UNet 학습:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe run_train_unet.py --config src/config/train_unet.yaml
```

여러 condition을 학습에 반영하려면 config 또는 CLI에서 조건 개수와 위치를 지정합니다.

```powershell
.\.venv\Scripts\python.exe run_train_unet.py `
  --config src/config/train_unet.yaml `
  --num-conditions 3 `
  --condition-axes z y x `
  --condition-slice-indices 12 20 28
```

DDP 학습은 `torchrun`으로 실행합니다. GPU 서버에서 사용하는 것을 전제로 합니다.

```powershell
$env:PYTHONPATH = "src"
torchrun --nproc_per_node 4 run_train_unet.py --config src/config/train_unet.yaml
```

512x512 crop으로 512x512x512 후보를 만들려면 학습과 inference 모두에서 slice index 범위를 감당할 수 있게 `max_slices`를 512 이상으로 맞춰야 합니다.

## 예측

예측은 `src/inference`의 함수 API를 사용합니다. 단일 조건은 `predict`, 여러 조건은 `predict_many`, 큰 crop 기반 scale-up은 `predict_scale_up`을 씁니다.

아래 예시는 64x64 조건 이미지를 넣어 64x64x64 후보를 만드는 최소 코드입니다.

```python
import numpy as np
import torch
from PIL import Image

from build import load_unet_checkpoint
from inference import ConditionSpec, predict_scale_up
from models import CustomVAE, DDPM, SliceConditionedTimeUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = CustomVAE(latent_ch=4).to(device).eval()
vae_ckpt = torch.load("microlad-anode/vae_anode.pth", map_location=device)
vae.load_state_dict(vae_ckpt["vae"] if "vae" in vae_ckpt else vae_ckpt)

unet = SliceConditionedTimeUNet(latent_ch=4, max_slices=64).to(device).eval()
checkpoint = torch.load("output/slice_conditioned/weights/last.pth", map_location="cpu")
load_unet_checkpoint(unet, checkpoint)

image = Image.open("data/images/sample_01.png").convert("L").resize((64, 64))
condition = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None, None].to(device)

ddpm = DDPM(timesteps=1000, device=device)
result = predict_scale_up(
    vae=vae,
    unet=unet,
    ddpm=ddpm,
    conditions=[ConditionSpec(condition=condition, axis=0, slice_index=12)],
    output_size=64,
    latent_ch=4,
    device=device,
)

volume = result["volume"]
print(volume.shape)
```

512x512 crop을 그대로 조건으로 쓰려면 `condition` shape을 `[1, 1, 512, 512]`로 만들고 `output_size=512`를 넘깁니다. 이 경우 메모리 사용량이 크므로 GPU 환경에서 실행하는 것을 권장합니다.

## 노트북

자주 보는 순서는 다음과 같습니다.

- `notebooks/00_patch_dataset.ipynb`: 원본 이미지에서 랜덤 crop을 뽑는 데이터셋 확인
- `notebooks/04_slice_conditioned_train_step.ipynb`: 조건부 학습 step 확인
- `notebooks/07_generate_pipeline.ipynb`: weight 로드 후 생성 흐름 확인
- `notebooks/09_scale_up_smoke.ipynb`: scale-up API smoke test
- `notebooks/10_reference_generate_smoke.ipynb`: 레퍼런스 weight 기반 생성 smoke
- `notebooks/11_microstructure_viewer.ipynb`: 생성된 TIFF volume slice viewer

노트북은 검증용으로만 짧게 유지합니다. 반복 검증은 `tests`에 둡니다.

## 테스트

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

현재 테스트는 데이터셋, VAE, DDPM/UNet, loss, trainer, inference API를 확인합니다.
