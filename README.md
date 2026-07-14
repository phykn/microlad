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

## 테스트

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## 참고

- [Original repository](https://github.com/KangHyunL/microlad)
- [MicroLad paper](https://arxiv.org/abs/2508.20138)
