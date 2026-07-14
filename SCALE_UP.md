# Scale-up 구현 및 검증 메모

이 문서는 scale-up의 구현 상태와 아직 실제 volume으로 확인해야 할 품질 위험을 기록한다. 코드 구현과 실제 128³ 품질 검증을 구분한다.

## 반영된 구조

- 큰 volume 전체가 하나의 3D latent residual을 공유한다. 2D phase slice를 순차 덮어쓰지 않는다.
- SDS와 descriptor는 세 축을 순회하는 학습 크기 latent crop에서 계산한다.
- initial large L-MPDD와 `checkpoint_every` 중간 latent를 모두 후보로 보존한다.
- phase ID가 아니라 latent만 최적화하므로 3상 이상의 ordinal 중간 phase 문제가 없다.
- 원본 latent preservation과 세 축 latent-gradient continuity를 함께 적용한다.
- 모든 후보를 동일한 interpolation·tri-axis consensus로 soft probability decode한다.
- anchor loss도 최종 tiled tri-axis consensus와 같은 좌표·확률을 사용한다. 매 step에는
  앵커 하나의 32×32 이하 영역을 순환해 GPU 메모리를 제한한다.
- scale Refine도 `strength`와 `anchor_strength`를 적용하고, calibration 변경량을 실제로 측정한다.
- calibration은 anchor target을 복사하지 않고 calibration 직전 모델 label만 보호한다.
- 합격 후보는 anchor의 미세한 수치 차이보다 morphology 기준으로 먼저 선택한다.

## 남은 품질 위험

- 단일 2D reference만으로 올바른 3D connected component나 percolation을 정의할 수 없다.
- geometric consensus가 얇은 희소상과 좁은 neck을 약화할 수 있다.
- crop 기반 stochastic guidance가 매우 큰 volume 전체를 충분히 방문하려면 steps와 batch size가 커져야 한다.
- 3D topology 진단은 기록하지만 아직 후보 gate나 학습 target은 아니다.

## 2026-07-14 실행 기록

- RTX 2060 6GB에서 80³ minimal regression은 623초에 완료됐고 peak memory는 2GB 미만이었다.
- near-duplicate rate와 repeat streak는 세 축 모두 0이었고 boundary jump는 6.4–7.3%였다.
- 첫 실행은 latent anchor MSE 버전이어서 anchor mismatch 27.9–29.3%, calibration 변경률 27.4%로 실패했고 initial L-MPDD 후보가 선택됐다.
- 이 결과로 latent 일치가 decoded anchor 품질을 보장하지 않는 원인을 확인해, scale anchor를 지정 축의 decoded categorical NLL로 교체했다.
- 교체 후 500-step 재실행은 step 400을 선택했고 calibration 변경률은 23.9%로 낮아졌지만, anchor mismatch는 29.1–29.9%여서 여전히 실패했다.
- near-duplicate rate와 repeat streak는 두 실행 모두 세 축 0이었고, 재실행 boundary jump는 최대 7.55%였다.
- perpendicular 두 축 contribution을 포함한 exact tri-axis anchor patch loss를 반영했다.
  full 80³ anchor 두 장을 매번 전부 미분하면 약 3–6초/step과 6GB가 필요해, 같은
  loss를 앵커·영역별로 순환 샘플링하도록 구현했다. 측정값은 약 0.3초/step,
  1.85GB다. anchor target을 voxel에 복사하는 방식은 사용하지 않는다.
- 같은 80³ initial latent의 200-step 비교에서 learning rate 0.0003은 28.2–29.7%,
  0.001은 24.6–25.8%, 0.003은 18.1–19.3% mismatch였다. 시간과 메모리는 같아
  scale 기본값을 0.003으로 조정했다.
- exact tri-axis loss와 learning rate 0.003의 500-step 실행은 step 500, Refine 0을
  선택했다. anchor mismatch는 8.6–9.2%로 VAE 단독 복원 9.1% 수준까지 내려갔다.
  near-duplicate와 repeat streak는 세 축 모두 0, boundary jump는 최대 7.47%,
  elapsed 341초, peak memory 약 2GB였다.
- 같은 실행의 calibration 변경률은 20.4%여서 5% budget은 통과하지 못했다. 원인은
  목표 phase fraction 약 29/13/58%에 비해 large L-MPDD의 축별 decoder 출력부터
  약 19/6.5/74%로 희소상을 적게 생성하는 것이다. geometric·power·arithmetic
  consensus 비교만으로는 이 축별 편향이 해결되지 않았다.
- exact consensus patch의 EMA fraction loss와 calibration pseudo-label loss도
  시험했지만 각각 anchor 17–18%, calibration 23–27%로 악화되어 채택하지 않았다.
  기존 categorical crop KL이 현재 가장 안정적이다.
- RTX 2060에서 full 128³ 기본 sweep과 100-step 축소 sweep은 모두 30분 실행 제한을 넘었다. OOM은 발생하지 않았다.

## 최종 평가 지표

- near-duplicate slice rate와 최대 반복 streak
- 세 축 boundary jump, transition, run-profile
- phase별 3D connected component 수
- phase·축별 percolation
- phase별 3D Euler density
- anchor mismatch와 calibration 변경률
- 목표 phase fraction 오차

앵커 mismatch와 phase fraction은 gate로 취급한다. gate를 통과한 후보 사이에서는 near-duplicate, 반복 streak, boundary, run-profile, transition을 먼저 비교한다. 3D topology는 목표값이 없는 상태에서 일률적으로 좋고 나쁨을 정하지 않고 진단값으로만 기록한다.

## 메모리 모드

- 3D latent residual 자체는 전체 volume에 하나만 두므로 block seam이 생기지 않는다.
- `scale.batch_size`는 한 guidance step에서 평가할 latent crop 수를 정한다.
- `scale.decode_batch_size`가 숫자이면 L-MPDD tile, anchor loss, final decode,
  VAE Refine을 나눠 처리한다.
- `scale.decode_batch_size: null`이면 큰 GPU에서 각 decode 단계를 한 batch로 처리한다.
- full/batched decode는 동일한 interpolation과 probability consensus를 사용한다.

## 완료 조건

- [x] scale guidance가 실패해도 initial large L-MPDD 후보로 복귀할 수 있다.
- [x] 3상 이상에서 ordinal 중간값을 최적화하지 않는다.
- [x] full/batched decoder가 같은 probability 결과를 낸다.
- [x] base-size와 같은 soft anchor·fraction calibration 계약을 사용한다.
- [x] sampled exact tri-axis anchor patch가 final decoder의 동일 영역과 일치한다.
- [x] 실제 80³ anchor mismatch를 12.5% 이내로 낮춘다.
- [ ] 실제 128³에서 모든 축의 near-duplicate slab와 시각적 seam이 없는지 확인한다.
- [ ] large L-MPDD의 희소 phase 편향을 줄여 80³ calibration 변경률을 5% 이내로 낮춘다.
- [ ] 실제 128³에서 anchor mismatch와 calibration 변경률을 확인한다.
- [ ] 실제 128³에서 component, percolation, Euler 지표를 baseline과 비교한다.
