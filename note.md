# 3D 조건부 복원 실험 노트

마지막 갱신: 2026-07-13

## 목표와 판정 순서

1. 지정한 위치의 XY 단면이 조건 이미지와 **유사**해야 한다. 조건 이미지를 그대로 복사하지는 않는다.
2. XY뿐 아니라 XZ/YZ 단면도 조건 이미지와 비슷한 2D 형태를 가져야 한다.
3. 인접 단면은 연속적으로 변해야 하며, 앵커에서 3칸을 벗어난 지점에서 갑자기 바뀌면 실패다.
4. 같은 XY가 z 방향으로 반복되어 XZ/YZ에 긴 기둥이나 띠가 생기면 실패다.
5. 시각적 형태를 먼저 보고, 그다음 앵커 오차·축 전이율·상분율을 확인한다. 수치만 좋아지고 파편이나 잡음이 늘면 채택하지 않는다.

초기 고정 진단 조건 이미지의 참고값:

- 상분율: `[0.3540, 0.1062, 0.5398]`
- 2D 이웃 전이율: `0.2228`

고정 seed를 완료 조건으로 사용하지 않으므로 위 값은 과거 sweep 비교용일 뿐이다. 실제 실행에서는 그 실행이 선택한 조건 이미지의 상분율과 형태를 다시 계산해 판정한다.

## 현재 채택한 기반

- categorical VAE의 확률 출력을 사용한다.
- 2D 한 장을 세 축에 독립적으로 복원하지 않고, 하나의 공유 3D generator가 만든 volume의 XY/XZ/YZ **모든 64개 단면**을 같은 2D critic으로 학습한다.
- 기본 texture 학습에는 조건 이미지의 회전·반사·주기 이동본을 사용한다. categorical VAE와 2D diffusion이 만든 8개 단면은 조건 이미지와 같은 상분율로 보정한 뒤 `10%`만 섞는다.
- texture 학습 중 `3000/4000/5000` step과 hybrid `+500/+1000` step을 morphology 점수로 후보화한다. 한 번의 GAN 종점이 불안정해도 좋은 중간 상태를 버리지 않기 위해서다.
- 각 후보에서 먼저 `4³` spatial noise를 조건 이미지에 맞추고, 그 다음 critic을 고정한 채 generator와 noise를 짧게 미세조정한다.
- 앵커를 voxel에 직접 복사하거나 ±몇 칸의 slab로 고정하지 않는다. 조건은 generator의 공간장 전체를 통해 전달되므로 ±3 밖에 별도의 경계가 없다.
- 실행마다 정확히 같은 volume을 다시 만드는 것은 목표가 아니다. 매 실행 결과 자체가 앵커 유사도·세 축 morphology·국소 연속성 기준을 통과하는지를 판정한다.

현재 가장 좋은 진단 결과(저장된 hybrid 6000-step 상태):

- 앵커 mismatch: `7.37%` — 조건과 유사하지만 동일하지 않음
- 상분율: `[0.3583, 0.1019, 0.5399]`
- 축 전이율: `[0.1867, 0.1883, 0.1832]`
- lag-3: `[0.3876, 0.3932, 0.3859]`
- 축별 run-profile MAE: `[0.0277, 0.0325, 0.0296]`
- 앵커 주변 z-boundary profile: `[0.2007, 0.1990, 0.1877, 0.2070, 0.1667, 0.1880, 0.1682, 0.1738, 0.1816, 0.2063]`
- 시각 판정: z=26..38이 점진적으로 변하고, ±3 밖 cutoff가 없으며, 세 축 모두 조건과 비슷한 blob morphology를 보임
- 남은 차이: 회색 phase의 2D Euler 수가 조건 약 `66`보다 생성 약 `40~41`로 낮아 작은 독립 domain은 아직 부족함

## 실험 기록

| 실험 | 주요 설정/결과 | 판정 | 이유 |
|---|---|---|---|
| 원형에 가까운 SliceGAN, anchor refs | 4³ noise, 모든 64개 단면/축, WGAN-GP, 5000 step | 채택 | 3000~5000 step에서 세 축의 blob morphology와 연속성이 처음으로 함께 안정됨 |
| diffusion reference 50% 직접 학습 | categorical VAE+2D diffusion 단면 8개 | 폐기 | critic margin 약 `99`, 상 소실과 파편·띠 구조가 발생함 |
| anchor 90% + diffusion reference 10% hybrid | 좋은 anchor checkpoint에서 1000 step 추가 | 채택 | step 6000 상분율 `[0.3580, 0.1057, 0.5363]`, margin 약 `29`; anchor-only 형태를 보존하면서 diffusion prior를 약하게 반영함 |
| frozen generator에서 noise만 800 step 조건화 | baseline mismatch 약 `29%`, hybrid 약 `25%` | 중간 단계만 채택 | 3D 연속성은 보존하지만 지정 단면 유사도가 부족함 |
| frozen critic, generator+noise 조건 미세조정 | G lr `1e-5`, noise lr `2e-3`, critic `0.02`, phase `50`, 최대 500 step | 채택 | mismatch `7.37%`, 변경 voxel `12.1%`; 앵커를 복사하지 않고 전역 구조도 유지함 |
| 좋은 SliceGAN 결과에 기존 joint 후처리 | anchor mismatch 약 `8%` | 폐기 | z=35→36에 새 cutoff가 생겨 원래의 연속성을 손상함 |
| 조건 미세조정에 continuity weight `0.01` | 나머지는 채택 설정과 동일 | 폐기 | 조건 형태와 상분율 손실에 비해 경계 profile 개선이 없었음 |
| morphology 기반 multi-checkpoint 선택 | primary 3000/4000/5000, hybrid +500/+1000; 상분율·전이율·run profile + 조건·경계 판정 | 채택 | 다른 조건 crop에서도 단일 final checkpoint의 GAN 변동성을 줄이고 내부 품질 기준을 통과함 |
| hard anchor slab | 앵커 주변 반경 2를 직접 고정 | 폐기 | z=29/35 부근에서 고정 영역 밖 구조가 갑자기 깨짐 |
| Gaussian anchor latent spread | 중심 latent를 주변 z에 확산 | 폐기 | 비슷한 XY가 반복되어 XZ/YZ에 긴 기둥과 띠가 생김 |
| tri-plane consensus + center-only soft anchor | 위의 안정적 기준 결과 | 채택 | hard cutoff와 강한 축 비대칭을 가장 안정적으로 줄임 |
| transpose-conv online SliceGAN 형태 | 300 step | 폐기 | grid/checkerboard collapse, 앵커 mismatch 약 51% |
| Patch/WGAN guidance | 짧은 online 학습 | 기본값에서 비활성 | discriminator가 generator를 압도하고 형태가 불안정 |
| SWD texture guidance | multi-scale patch distribution | 폐기 | 큰 덩어리로 과도하게 평활화됨 |
| phase-interface guidance | phase pair 경계 통계 | 폐기 | 큰 domain을 만들고 조건 형태와 멀어짐 |
| categorical VAE 1회 3축 투영 | mismatch `13.96%`, 전이율 `[0.1677, 0.1751, 0.1786]` | 폐기 | 앵커와 상분율이 악화되고 XZ/YZ 큰 덩어리가 커짐 |
| 저노이즈 diffusion 후처리, t=25 | 변경량 1.5~9.8% sweep | 폐기 | 모양은 거의 그대로이고 강할수록 경계만 평활화됨 |
| 저노이즈 diffusion 후처리, t=100 | 변경량 2.0~10.9% sweep | 폐기 | XZ/YZ의 긴 구조가 줄지 않고 앵커 오차만 증가 |
| global transition weight=1.0 | 전이율 `[0.2097, 0.2139, 0.2145]`, mismatch `10.60%` | 폐기 | 목표 전이율에는 가까워졌지만 잔경계와 파편이 과도함 |
| transition=0.5, continuity=0.002, SA=1.0 | 전이율 `[0.1956, 0.2039, 0.2044]`, mismatch `10.18%` | 폐기 | 긴 덩어리는 줄었지만 XZ/YZ 파편화가 여전히 큼 |
| 낮은 SDS timestep 범위 | `t=10..200`, transition `0.35`, continuity `0.002`, SA `1.0`, anchor `0.075` | 이전 기준 | mismatch `8.69%`, 전이율 `[0.1913, 0.2031, 0.2051]`, 상분율 `[0.3521, 0.1005, 0.5474]`; transition=1의 파편화를 완화했지만 XZ/YZ가 XY만큼 둥글지 않음 |
| multi-scale run profile | transition을 끄고 길이 `2/4/8/16` run loss `0.25` | 현재 채택 | mismatch `7.01%`, 전이율 `[0.1769, 0.1882, 0.1865]`, 상분율 `[0.3589, 0.0976, 0.5434]`, 축별 run MAE `[0.0372, 0.0271, 0.0291]`; 파편과 앵커 오차가 줄었으나 XZ/YZ의 연결 topology는 조건과 여전히 다름 |
| phase별 Euler density | run profile에 Euler loss `1e-5` 추가 | 폐기 | mismatch `8.03%`, 상분율 `[0.3574, 0.0861, 0.5565]`; 회색 phase 목표 `66.4` 대비 세 축 약 `33`이고 시각적 연결망도 개선되지 않아 단일 topology 스칼라가 불충분함 |
| 직접 64³ voxel logits | Conv3D 보정장을 없애고 각 voxel을 직접 최적화, lr `0.02` | 폐기 | mismatch `0.00%`로 앵커를 그대로 복사했고, 중심 경계가 `0.5247`로 주변 중앙값 `0.2651`보다 급증했다. 전이율 `[0.2651, 0.2653, 0.2656]`, run MAE `[0.0572, 0.0519, 0.0546]`; 세 축 모두 심하게 파편화되어 3D 공간 정규화가 반드시 필요함 |
| 매 step 세 축 동시 slice batch | 총 batch 16을 XY/XZ/YZ에 나누어 매 step 동시에 최적화 | 폐기 | mismatch `7.25%`, 전이율 `[0.1737, 0.1844, 0.1850]`, run MAE `[0.0367, 0.0287, 0.0298]`; 기존 run-profile 결과와 시각적으로 거의 같아 축을 번갈아 갱신한 순서가 주원인은 아님 |
| 안정화 categorical slice critic, weight=0.02 | hard categorical 입력, spectral normalization, hinge loss, 60-step ramp | 보류 | 상분율 `[0.3553, 0.1024, 0.5422]`, mismatch `8.96%`, 전이율 `[0.1833, 0.1957, 0.1908]`; 회색 phase Euler가 약 `29~31`에서 `33~35`로 개선됐지만 critic 항이 평균 loss의 약 2%라 XZ/YZ 연결망 변화는 작음 |
| 안정화 categorical slice critic, weight=0.10 | 나머지는 weight `0.02`와 동일 | 폐기 | mismatch `14.53%`, 상분율 `[0.3266, 0.1139, 0.5594]`, run MAE `[0.0404, 0.0436, 0.0426]`; critic margin은 `0.27→0.25`로 거의 줄지 않은 채 다른 조건만 무너져 단순 가중치 증가는 해법이 아님 |
| 초기 L-MPDD logit strength=0.25 | critic `0.02`, 나머지는 동일 | 폐기 | mismatch `11.77%`, 전이율 `[0.1595, 0.1702, 0.1676]`, run MAE `[0.0497, 0.0437, 0.0442]`; 큰 매끈한 연결망으로 수렴하고 critic margin이 `0.94`로 커져 초기장 결합만 약화하는 것은 해법이 아님 |
| local categorical slice critic, RF=22, weight=0.02 | 46px critic에서 receptive field만 22px로 축소 | 보류 | mismatch `8.57%`, 상분율 `[0.3583, 0.1021, 0.5396]`, run MAE `[0.0360, 0.0278, 0.0285]`, margin `0.19`; 수치 안정성과 회색 Euler `34~36`은 개선됐지만 XZ/YZ 시각 변화는 아직 작음 |
| local categorical slice critic, RF=22, weight=0.05 | 나머지는 RF=22, weight `0.02`와 동일 | 폐기 | mismatch `10.40%`, 상분율 `[0.3388, 0.1118, 0.5495]`, run MAE `[0.0395, 0.0357, 0.0336]`; 회색 Euler는 `36~37`로 올랐지만 조건·상분율 손실과 XZ/YZ 연결망을 함께 해결하지 못함 |
| multiscale logit pyramid | `8³+16³+32³`, lr `0.01`, local critic `0.02` | 폐기 | mismatch `12.33%`, 상분율 `[0.3355, 0.0988, 0.5658]`, run MAE `[0.0384, 0.0328, 0.0364]`; 작은 독립 domain은 늘었지만 조건 손실과 critic margin `0.73`이 커지고 전체 형태가 여전히 다름 |
| fixed-noise online generator, 600 step | 4³ noise, uniform-information generator, 전체 192 slice critic, anchor `0.25` | 폐기 | mismatch `8.50%`지만 상분율 `[0.3821, 0.0563, 0.5616]`, run MAE `[0.0545, 0.0375, 0.0939]`; XZ/YZ에 규칙적인 대각선·격자 패턴이 생겨 한 latent에 texture와 anchor를 동시에 강제하는 방식은 실패 |
| 2-stage online generator, 1200 step | random-noise texture `900` + frozen-generator latent condition `300` | 폐기 | 회색 phase가 `0%`로 소실되고 mismatch `54.05%`, critic margin `3.11`; generator가 critic에 완전히 압도되어 모든 축에 강한 주기 패턴이 생김 |
| image-space harmonization, anchor=0.15 | 4 sweep, t=`100`, blend=`0.25`, coarse latent anchor 미사용 | 폐기 | 전체 변화량 `9.18%`, mismatch `1.76%`로 앵커를 사실상 복사했고 run MAE가 약 `0.041~0.047`로 악화됨; 회색 Euler `38~39` 개선만으로는 채택 불가 |
| image-space harmonization, anchor=0.02 | 나머지는 anchor=`0.15`와 동일 | 폐기 | mismatch `9.35%`로 복사 문제는 해결했지만 전체 변화량 `8.99%`, run MAE 약 `0.042~0.047`; XZ/YZ morphology가 거의 개선되지 않아 반복 저노이즈 projection도 해법이 아님 |

현재 `03_predict.ipynb`의 선택값:

```text
slicegan_steps = 5000
slicegan_hybrid_steps = 1000
slicegan_condition_steps = 800
slicegan_finetune_steps = 500
```

## SliceGAN 원 구현에서 확인한 차이

- 원 논문은 한 생성 volume에서 XY/XZ/YZ의 **64개 단면 전부**를 critic에 전달한다. 최소 32개, 실제로는 64개 모두가 더 안정적이라고 보고한다.
- 공식 구현은 `4³` spatial noise, uniform-information transpose convolution generator, WGAN-GP, generator/discriminator lr `1e-4`, critic iteration `5`, 100 epochs를 사용한다.
- 논문에 보고된 학습 시간은 Titan Xp 기준 약 4시간이다. 따라서 앞선 300-step, 16-slice 보조 critic은 SliceGAN을 충분히 재현한 실험이 아니며, "짧은 GAN"의 한계로 해석한다.
- 참고: [SliceGAN 논문](https://arxiv.org/abs/2102.07708), [공식 구현](https://github.com/stke9/SliceGAN)

## 해석

- XZ에서 구조가 길어지는 직접 원인은 z 방향으로 같은 상 배치가 오래 유지되기 때문이다. 거의 같은 XY가 반복되면 XZ/YZ에서는 세로 기둥이나 긴 띠가 된다.
- 2D diffusion prior만으로도 원칙상 가능하지만, 세 축의 2D 조건을 **한 공유 3D voxel field에서 동시에 만족**시켜야 한다. 축별로 독립 생성한 뒤 합치는 방식은 이 일관성을 보장하지 못한다.
- 전이율 하나만 맞추면 모델은 작은 파편을 추가하는 쉬운 해를 선택한다. 따라서 전이율 개선은 시각적 domain 크기·형태와 함께 판단해야 한다.
- run profile은 연속 길이와 파편 문제를 개선하지만, 같은 길이 분포라도 분리된 둥근 영역과 서로 연결된 망은 구분하지 못한다.
- 기존 joint-SDS 경로의 병목은 `phase별 2D topology(연결 성분과 hole의 균형)`였고, SliceGAN의 공유 3D generator가 이를 크게 개선했다.
- 2D diffusion reference를 많이 섞으면 오히려 학습이 무너진다. 현재 diffusion은 새로운 texture 전체를 결정하는 teacher가 아니라 categorical 형태 다양성을 약하게 보완하는 prior로 사용한다.
- 조건 이미지는 한 voxel 면에 붙여 넣는 대상이 아니라, generator의 spatial noise와 weight가 만들어 내도록 유도하는 목표다. 그래서 중심 단면을 완전히 같게 만드는 것보다 `약 7~8% mismatch`를 허용하는 편이 조건성과 3D 자연스러움의 균형이 좋다.

## 다음 진행 규칙

- 한 번에 가설 하나만 바꾼다.
- 정확한 seed 재현은 요구하지 않는다. 비교가 필요한 단일 sweep 안에서는 같은 입력을 유지하되, 최종 판정은 각 실행 결과의 품질 기준으로 한다.
- 각 실행 뒤 이 표에 파라미터, 핵심 수치, 시각 판정, 채택 여부를 기록한다.
- 폐기한 방법은 새로운 근거가 없으면 다시 켜지 않는다.
- `04_scale_up.ipynb` 작업은 사용자가 다시 요청할 때까지 중단한다.

## 검증 상태

- `03_predict.ipynb`: SliceGAN 경로로 단순화하고 처음부터 끝까지 실행 완료 (`759.0초`, CUDA)
- 최종 노트북 실행: step `5500` 선택, 조건 후보 `3개` 비교, 앵커 mismatch `7.96%`
- 최종 노트북 실행 상분율: 조건 `[0.2578, 0.1270, 0.6152]`, 전체 `[0.2604, 0.1254, 0.6142]`
- 최종 노트북 실행 축 전이율: `[0.2000, 0.2104, 0.2019]`
- 최종 노트북 실행 z=26..38: boundary profile `[0.1836, 0.1829, 0.1943, 0.1929, 0.2185, 0.2249, 0.2485, 0.1787, 0.1814, 0.2195, 0.1826, 0.1604]`, 최대 인접 jump `0.0698`
- 최종 시각 판정: 중심 조건을 복사하지 않고 큰 domain 배치를 유지함. z±6은 누적해서 변하며 ±3 밖에 hard cutoff가 없음. XY/XZ/YZ 전역 몽타주는 모두 blob morphology를 유지하고 이전의 긴 기둥·강한 축 차이가 보이지 않음
- 노트북 JSON, cell id, 모든 코드 셀 문법 검증 완료
- 실제 Predictor 1-step smoke: `64³`, categorical `uint8`, OOM 없이 통과
- 저장 checkpoint 기반 전체 조건화/시각 검증: 통과
- multi-checkpoint 통합 Predictor 전체 실행: 내부 mismatch `8.03%`, phase MAE `0.00164`, 경계 표준편차 `0.02882`, 최대 국소 경계 jump `0.05029`; 내부 기준 통과
- focused tests: `39 passed, 23 subtests passed`
- 전체 테스트: `442 passed, 96 subtests passed`
