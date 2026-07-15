# Conditioned MicroLad

2D 미세구조 이미지로 VAE, latent diffusion, latent GAN을 학습하고 조건부 3D
categorical volume을 생성하는 프로젝트입니다.

## 설치

```powershell
python -m pip install -r requirements.txt
```

명령은 저장소 루트에서 실행합니다. 실제 데이터와 학습 checkpoint는 저장소에
포함되지 않습니다.

## 구조

```text
src/modeling/          모델 정의
src/pipeline/train/    VAE, Diffusion, GAN 훈련
src/pipeline/predict/  L-MPDD, Joint, Refine, scale-up 예측
src/app/               설정 로딩과 공개 API
config/                훈련·예측 설정
notebooks/             단계별 확인 예제
```

## 훈련

VAE를 먼저 학습한 뒤 Diffusion과 GAN을 학습합니다.

```powershell
python run_train_vae.py
python run_train_diffusion.py
python run_train_gan.py
```

- VAE: `config/vae.yaml`
- Diffusion: `config/diffusion.yaml`
- GAN: `config/gan.yaml`
- 출력: `run/<timestamp>/`

Diffusion과 GAN 설정의 `output.vae_run_dir`에는 학습된 VAE run 경로를 지정합니다.

## 3상 시뮬레이션 데이터

`simul/gen_data.py`는 `src`와 독립된 단순 구형 데이터 생성기입니다.
3D 배열에 `0=배경`, `1=작은 구`, `2=큰 구`를 배치하고, 볼륨은 다중 페이지
TIFF로, 모든 Z 단면은 학습용 categorical PNG로 저장합니다.

```powershell
.venv\Scripts\python.exe simul\gen_data.py
.venv\Scripts\python.exe simul\gen_data.py --num-volumes 20
```

기본값은 128³ 정답 볼륨 하나를 `data/gt/volume.tif`에 저장합니다. 학습 PNG는
양 끝의 노이즈성 단면을 8장씩 제외한 Z=8~119의 112장만 `data/train/`에
생성합니다. `config/vae.yaml`은 이 PNG의 라벨 0/1/2를 직접
읽도록 `segment: false`로 설정되어 있습니다. 기본 상 비율은 `sample.png`에 맞춘
`배경 22.9% / 작은 구 27.1% / 큰 구 50.1%`를 목표로 합니다. 큰 구를 먼저
무작위 X/Y 위치에서 아래로 떨어뜨려 골격을 만들고, 작은 구는 골격 내부의 충돌
없는 3D 공극에 침투시킵니다. 큰 구는 바닥이나 기존 구에 처음 닿는 Z 위치에
배치하며, 여러 낙하 후보 중 가장 깊은 위치를 선택하는 단순한 구름(rolling)도
적용합니다. 작은 구 반경은 항상 큰 구
반경의 절반이며, 기본 반경은 큰 구 20 voxel, 작은 구 10 voxel입니다. 비겹침
때문에 목표 밀도에 못 미쳐도 격자 배치나 더 작은 보충 구는 사용하지 않습니다.
여러 볼륨을 만들 때는 `volume_000.tif`, `volume_001.tif`와
`volume_000_z_008.png` 같은 이름을 사용하며, 볼륨마다 seed를 1씩 증가시킵니다.

## 예측

`config/predict.yaml`의 VAE, Diffusion, GAN run 경로와 예측 조건을 설정합니다.

```python
from src.app.api import PredictOptions
from src.app.runtime import load_predict_config, load_predictor

model_runs, config = load_predict_config("config/predict.yaml")
predictor = load_predictor(**model_runs, device="cuda")
options = PredictOptions(num_phases=predictor.vae.num_phases, **config)
volume, stats = predictor.predict(options)
```

기본 흐름은 `L-MPDD → Joint → optional Refine → categorical volume`입니다.

## Critic fake data

기존 VAE와 Diffusion checkpoint를 고정한 채 L-MPDD 3D latent를 미리 생성합니다.
모델 학습은 수행하지 않습니다.

```powershell
.venv\Scripts\python.exe gen_fake.py --check
.venv\Scripts\python.exe gen_fake.py --device cuda
```

설정은 `config/gen_fake.yaml`에 있으며 결과는 저장소 루트의 `fake/` 아래에
`00000.pt`, `00001.pt` 형태의 `[C, D, H, W]` latent로 저장됩니다. Critic 학습에서는
`config/gan.yaml`의 `data.fake_dir`을 읽어 XY·XZ·YZ latent 단면을 균형 추출하고,
VAE로 phase probability 이미지에 디코딩한 뒤 image critic에 입력합니다. Generator는
기존처럼 latent를 생성하며 image critic의 gradient로 함께 학습됩니다.

## 테스트

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## 참고

- [Original repository](https://github.com/KangHyunL/microlad)
- [MicroLad paper](https://arxiv.org/abs/2508.20138)
