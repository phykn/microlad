# MicroLad 수학 감사 설계

## 목표

현재 구현의 수학적 타당성을 논문 원문, 코드, 테스트, 작은 합성 수치 실험으로 교차 검증한다. 논문과 다른 구현을 자동으로 오류로 취급하지 않고, 사용자가 추가한 스케일업과 최근의 의도적 개선을 독립적인 설계 선택으로 평가한다. 감사 결과는 이후 코드 스타일, 이름, 파일 및 폴더 구조를 리팩터링하는 우선순위와 근거가 된다.

## 범위와 전제

- 공개 Python API, 설정 키, 체크포인트 형식의 하위 호환성은 보장하지 않아도 된다.
- 논문 재현보다 수학적으로 타당하고 명확한 구현을 우선한다.
- 현재 코드에 이미 들어간 개선은 논문과 다르다는 이유만으로 되돌리지 않는다.
- 스케일업은 사용자가 추가한 핵심 기능이며, 삭제하거나 축소하지 않는다.
- 감사에는 작은 CPU 합성 실험을 포함하지만 실제 모델 재학습과 품질 벤치마크는 포함하지 않는다.
- 감사 단계에서는 제품 코드를 수정하지 않는다. 발견 사항과 재현 실험을 먼저 문서화한다.
- 작업 트리의 기존 미커밋 변경은 보존하며 감사 대상에는 현재 상태 그대로 반영한다.

## 감사 접근법

감사의 기본 단위는 수식-코드 추적 항목이다. 각 항목은 다음 순서로 기록한다.

1. 기준 수식 또는 요구 불변조건
2. 관련 논문 페이지와 코드 위치
3. 현재 구현의 계산 과정과 tensor shape
4. 논문 대비 차이와 의도적 개선 여부
5. 작은 합성 입력으로 실행한 검증
6. 판정, 위험도, 개선 권고

판정은 다음 다섯 종류를 사용한다.

- `정상`: 기준 수식과 구현이 일치하거나 수학적으로 동치다.
- `의도적 개선`: 논문과 다르지만 근거와 장점이 명확하다.
- `의심`: 수학적으로 가능하지만 근거 또는 검증이 부족하다.
- `오류`: 수식, 단위, gradient, 정규화 또는 축 처리가 잘못됐다.
- `구조 위험`: 현재 계산은 맞지만 중복이나 결합 때문에 변경 시 오류 가능성이 높다.

위험도는 `치명적`, `높음`, `중간`, `낮음`으로 나눈다. 위험도는 결과 왜곡의 크기, 발생 범위, 탐지 난이도, 후속 모듈에 미치는 영향을 함께 고려한다.

## 감사 영역

### 1. 데이터 표현과 phase semantics

정수 phase label, categorical logits, soft phase probability, 연속 phase value의 의미를 분리해서 추적한다. segmentation, VAE 출력, descriptor 계산, anchor 손실이 동일한 phase convention과 범위를 사용하는지 확인한다.

### 2. VAE

categorical reconstruction loss, KL divergence의 reduction과 beta 가중치, reparameterization, encoder 및 decoder shape, latent scaling을 검사한다. phase-logit reconstruction은 현재 구현의 의도적 개선 후보로 평가한다.

### 3. LDM과 DDPM

forward noising, noise-prediction objective, reverse mean, posterior variance, timestep 표본 범위와 경계를 검사한다. 코드의 schedule tensor가 같은 indexing convention을 사용하는지 확인한다.

### 4. 3D 재구성

세 Cartesian 축의 slice 추출과 복원, latent update, 축별 결과 평균, anchor 조건, decode 및 refinement를 검사한다. 좌표 변환과 tensor 축 순서가 실제 같은 voxel을 가리키는지 확인한다.

### 5. SDS와 목적함수

SDS pseudo-gradient, timestep weighting, gradient 차단 위치, volume fraction, two-point correlation, relative surface area, differentiable FEM diffusivity를 검사한다. 논문의 표시 수식과 실제 pseudo-gradient surrogate가 수학적으로 같은 업데이트 방향을 만드는지 별도로 검증한다.

### 6. 스케일업

tile 좌표와 coverage, overlap, Hann weighting, tiled denoising, tiled decoding, local SDS prior, global descriptor objective, 큰 anchor의 좌표 정렬을 검사한다. 스케일업은 논문과의 불일치가 아니라 별도 확장 설계로 평가한다.

## 합성 수치 실험

실험은 고정 seed와 작은 CPU tensor를 사용하며, 가능한 경우 float64로 계산한다.

- Gradient: `torch.autograd.gradcheck` 또는 중앙 유한차분으로 gradient 방향과 크기를 비교한다.
- VAE와 DDPM: 닫힌형 계산과 직접 비교하고 첫 timestep과 마지막 timestep을 포함한다.
- Phase descriptor: 상수상, 정확한 반반 분할상, checkerboard처럼 기대값을 계산할 수 있는 입력을 사용한다.
- TPC/S2: 평행 이동, 회전, 축 교환 및 주기 경계 가정에 대한 불변성을 검사한다.
- Surface area: 경계 padding과 해상도 변화에 따른 정규화 및 단위 변화를 검사한다.
- Diffusivity: 균질장, 완전 차단장, 직선 통로의 기대 해와 비교한다.
- SDS: 논문의 제곱 손실에서 유도한 gradient와 현재 surrogate loss gradient의 방향과 상대 scale을 비교한다.
- 스케일업: 단일 tile과 tiled 결과의 동등성, overlap 변화의 seam, tile 순서 불변성, local 및 global objective의 gradient scale을 비교한다.
- Anchor: image와 latent 해상도 변환 후에도 좌표가 같은 실제 위치를 가리키는지 검사한다.

## 기존 테스트의 역할

전체 테스트를 먼저 실행해 현재 기준선을 기록한다. 기존 테스트는 회귀 방지 근거로 사용하지만, 통과 사실만으로 수학적 정확성을 판정하지 않는다. 감사 중 만든 실험은 독립 실행 가능한 audit test 또는 script로 남겨 이후 리팩터링의 characterization test로 재사용할 수 있게 한다.

## 산출물

주 감사 보고서는 `docs/audit/` 아래 한국어 Markdown으로 작성한다. 보고서에는 다음 내용이 포함된다.

- 전체 요약과 위험도별 발견 사항
- 영역별 수식-코드 추적표
- 각 합성 실험의 입력, 기대 결과, 실행 결과
- 유지할 의도적 개선 목록
- 수정할 수학 오류와 구조 위험 목록
- 의존관계에 따른 리팩터링 권장 순서
- 실제 재학습이나 대규모 benchmark가 필요한 잔여 불확실성

감사 보고서가 승인되면 발견 사항을 독립적으로 검증 가능한 리팩터링 단계로 나눈다. 기본 순서는 데이터 표현과 공통 수학 primitive, VAE/DDPM, descriptor와 FEM, 3D 재구성, 스케일업 orchestration, 공개 entrypoint와 문서 순이다. 실제 우선순위는 감사 결과의 위험도와 의존관계로 확정한다.

## 완료 기준

- 여섯 감사 영역의 모든 핵심 수식과 불변조건이 코드 위치에 연결되어 있다.
- 각 `오류` 또는 `의심` 판정에는 재현 가능한 합성 실험이나 명시적인 정적 증명이 있다.
- 논문과 다른 구현은 `의도적 개선`, `오류`, `미검증 확장` 중 하나로 구분되어 있다.
- 스케일업의 tile, objective, anchor 불변조건이 독립적으로 평가되어 있다.
- 리팩터링 순서가 발견 사항의 위험도와 의존관계로 설명되어 있다.
- 제품 코드는 감사 단계에서 변경되지 않았다.
