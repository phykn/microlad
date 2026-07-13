# Scale-up 개선 방향

현재 우선순위는 `03_predict.ipynb`의 base-size 조건부 3D 생성 완성이다. 이 문서는 scale-up 후속 작업의 범위와 순서를 기록하며, 아래 항목은 아직 품질이 검증된 구현으로 간주하지 않는다.

## 현재 한계

- 큰 volume은 2D slice를 순차적으로 수정하므로 다른 축에서 이미 좋아진 구조를 다시 훼손할 수 있다.
- scale guidance 전의 large L-MPDD 결과와 중간 checkpoint가 최종 후보로 보존되지 않는다.
- 3상 이상에서도 단일 float phase label을 직접 최적화해 존재하지 않는 중간 phase 값을 만들 수 있다.
- base Joint의 3D continuity와 원본 latent preservation에 대응하는 제약이 부족하다.
- base와 scale의 Refine·calibration 후보 의미가 완전히 같지 않다.
- 단면 texture는 좋아도 3D connected component, percolation, 좁은 neck과 같은 topology는 보장하지 않는다.

## 구현 우선순위

1. 초기 large L-MPDD volume을 후보 0으로 보존한다.
2. scale guidance의 일정 간격 checkpoint를 모두 후보로 남긴다.
3. anchor와 phase fraction은 통과 조건으로 사용하고, 통과 후보는 3D morphology 기준으로 선택한다.
4. scalar phase label 대신 `[P, D, H, W]` categorical logits 또는 probability simplex를 최적화한다.
5. 개별 2D slice 덮어쓰기 대신 overlap되는 3D latent block을 공동 최적화한다.
6. 각 block에 원본 latent preservation과 세 축 continuity를 함께 적용한다.
7. base와 scale decoder가 동일한 interpolation·tri-axis consensus 의미를 사용하도록 유지한다.
8. base와 scale의 Refine·calibration·candidate selector 계약을 통일한다.

## 최종 평가 지표

- near-duplicate slice correlation과 최대 반복 streak
- 세 축 boundary jump, transition, run/chord-length distribution
- 3D connected component 수와 크기 분포
- 축별 percolation
- 3D Euler characteristic
- anchor mismatch와 calibration 변경률
- 목표 phase fraction 오차

앵커 mismatch와 phase fraction은 hard gate로 취급한다. gate를 통과한 후보 사이에서는 3D topology와 연속성을 먼저 비교하고, 앵커의 작은 수치 차이는 마지막 tie-breaker로만 사용한다.

## 메모리 모드

- 메모리가 충분하면 전체 large latent 또는 큰 3D block을 한 번에 처리한다.
- 메모리가 부족하면 overlap block과 gradient checkpointing을 사용하되, block 경계에서 동일한 loss와 blending 규칙을 유지한다.
- 두 모드는 메모리 사용만 달라야 하며 생성 의미와 후보 평가 기준은 같아야 한다.

## 완료 조건

- scale guidance가 실패해도 초기 large L-MPDD 후보로 복귀할 수 있다.
- 3상 이상에서 ordinal 중간값을 최적화하지 않는다.
- 모든 축에서 near-duplicate slab와 block seam이 없다.
- base-size와 같은 anchor·fraction 계약을 만족한다.
- 128³ 이상 실제 volume에서 단면 시각 품질뿐 아니라 3D topology 지표도 회귀 검증한다.
