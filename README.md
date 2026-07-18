# Microlad

2D 단면 확산 모델로 범주형 3D 미세구조를 생성합니다. 하나의 U-Net이 축
`0`, `1`, `2`를 번갈아 처리하며 상 비율과 soft anchor 조건을 지원합니다.

## 실행

```powershell
python -m pip install -r requirements.txt
python gen_data.py
python run_train.py
```

- 데이터 생성 설정: `config/simul.yaml`
- 학습 설정: `config/model.yaml`
- 예측 설정: `config/predict.yaml`
- 체크포인트: `run/<timestamp>/weight/mpdd/`

## 데이터

`config/model.yaml`에서 축별 이미지 폴더를 지정합니다.

```yaml
data_dir:
  0: ../data/generated/train/0
  1: ../data/generated/train/1
  2: ../data/generated/train/2
```

이미지는 `0`부터 `num_phases - 1`까지의 정수 상 라벨을 사용해야 합니다.
세 축은 이미지 개수와 관계없이 같은 비율로 학습됩니다.

## 예측

```python
from src.predict import load_predict_config, load_predictor

cfg = load_predict_config("config/predict.yaml")
pred = load_predictor(cfg.run_dir)
opts = cfg.make_options(pred)
vol, stats = pred.predict(opts)
```

`vol`은 `[D, H, W]` 형태의 `uint8` 텐서입니다. 앵커는 `AnchorSlice` 목록으로
`predict`에 전달할 수 있으며, 생략하면 빈 앵커 조건으로 생성합니다.

## 동작

- 축 조건과 앵커 인코더는 항상 모델에 포함됩니다.
- 상 비율 조건은 classifier-free guidance를 지원합니다.
- DDPM, DDIM, harmonization과 overlapping tile 생성을 지원합니다.
- 큰 입자는 `big_elongation: 1.0`일 때 구이고 작은 입자는 항상 구입니다.

예제는 `notebooks/`에 있습니다.

## 테스트

```powershell
.venv\Scripts\python.exe -m pytest -q
```
